import os
import random
import time
import PIL.ImageOps
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import model_gan

print(torch.__version__)
print(torchvision.__version__)

# 定义一些超参
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]
test_batch_size = 25        # 测试时batch_size
result_path = "./GAN_output"
save_image_name = "gan_images"
model_path = "model"      # 存放模型的位置
location_index_name = 'location_index.txt'
image_size = (224, 224)       # 图片尺寸
latent_z_dim = 100
img_shape = (3, 224, 224)
unknown_type_index = [2, 6, 7, 8]      # 未知缺陷类别在标签中的索引，如：污点2位置为2


def open_file(filepath):
    with open(filepath, 'r') as f:
        file_data = f.readlines()
        data = []
        for row in file_data:
            tmp_list = row.split('\t')
            data.append(tmp_list)
    return data


class SiameseNetworkDataset(Dataset):
    def __init__(self, dataset_path, sample_A, sample_B, transform, should_invert=True):
        self.dataset_path = dataset_path
        self.sample_A = sample_A
        self.sample_B = sample_B
        self.img_list = sample_B.copy()
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, item):
        img0_list = random.choice(self.sample_A)
        img1_list = random.choice(self.img_list)
        self.img_list.remove(img1_list)
        # print(len(self.img_list))

        img0_path = os.path.join(self.dataset_path, img0_list[0])
        img1_path = os.path.join(self.dataset_path, img1_list[0])

        label_ = []
        for i in range(1, 10):
            label_.append(int(img1_list[i]))
        label1 = torch.tensor(label_, dtype=torch.float32)

        if img1_list[1] == "1":
            gen_label = 1
        else:
            gen_label = 0

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label1, gen_label, img1_list[0]

    def __len__(self):
        return len(self.sample_B)


# 定义Dataloader
dataset_path = './ImageDataset'
sample_A_dir = './dataset_txt/污点-2/sample_A.txt'     # 样本A文件
known_test_dir = './dataset_txt/污点-2/test.txt'       # 验证集文件
unknown_test_dir = './dataset_txt/污点-2/unknown.txt'
unknown_test_dir1 = './dataset_txt/unknown/毛刺/unknown.txt'
unknown_test_dir2 = './dataset_txt/unknown/裂纹/unknown.txt'
unknown_test_dir3 = './dataset_txt/unknown/其他/unknown.txt'
sample_A_list = open_file(sample_A_dir)
test_list = open_file(known_test_dir)
unknown_test_list = open_file(unknown_test_dir)
unknown_test_list1 = open_file(unknown_test_dir1)
unknown_test_list2 = open_file(unknown_test_dir2)
unknown_test_list3 = open_file(unknown_test_dir3)
test_list.extend(unknown_test_list)
test_list.extend(unknown_test_list1)
test_list.extend(unknown_test_list2)
test_list.extend(unknown_test_list3)
transform = transforms.Compose([transforms.Resize(image_size),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

test_siamese_dataset = SiameseNetworkDataset(dataset_path=dataset_path,
                                             sample_A=sample_A_list,
                                             sample_B=test_list,
                                             transform=transform,
                                             should_invert=False)
test_dataloader = DataLoader(test_siamese_dataset,
                             shuffle=True,
                             num_workers=1,
                             batch_size=test_batch_size)

# 定义模型且移至GPU
C_net = torch.nn.DataParallel(model_gan.SiameseNetwork(num_classes=9), device_ids=device_ids).cuda()
G_net = torch.nn.DataParallel(model_gan.Generator(img_shape, latent_z_dim), device_ids=device_ids).cuda()
D_net = torch.nn.DataParallel(model_gan.Discriminator(img_shape), device_ids=device_ids).cuda()

# 加载预训练的权重及偏置等参数和网络
weight_path_C = os.path.join(result_path, model_path, 'C_net(unknown_s2).pth')
weight_path_G = os.path.join(result_path, model_path, 'G_net(unknown_s2).pth')
weight_path_D = os.path.join(result_path, model_path, 'D_net(unknown_s2).pth')
C_net.load_state_dict(torch.load(weight_path_C))
G_net.load_state_dict(torch.load(weight_path_G))
D_net.load_state_dict(torch.load(weight_path_D))

location_index_path = os.path.join(result_path, location_index_name)
with open(location_index_path, "a") as location_index_file:
    now = time.strftime("%c")
    location_index_file.write('\n=============== Saving location index of abnormal images(%s)(test) ===============\n' % now)

num_unknown = torch.zeros(len(unknown_type_index))
unknown_correct = torch.zeros(len(unknown_type_index))
unknown_recall = torch.zeros(len(unknown_type_index))
pred_true = 0
preds = 0
gen_labels = 0
# 开始测试
for i, data in enumerate(test_dataloader):
    img0, img1, label, gen_label, img_name = data
    img_name = list(img_name)
    img0, img1, label, gen_label = img0.cuda(), img1.cuda(), label.cuda(), gen_label.cuda()
    result, d_features = C_net.forward(img0, img1)

    z = torch.FloatTensor(torch.normal(0, 1, (len(label), latent_z_dim))).cuda()
    gen_imgs = G_net.forward(d_features, z)
    score = D_net.forward(gen_imgs)
    pred = torch.as_tensor(score.ge(0.5), dtype=torch.int64)

    save_image(gen_imgs.data[:len(label)], os.path.join(result_path, save_image_name, 'test', '%d.png' % i),
               nrow=5, normalize=True)
    with open(location_index_path, "a") as location_index_file:
        location_index_file.write('\nThe  location index of abnormal images %d:\n' % i)
        for j in range(len(label)):
            if label[j][0] != 1:
                row = j / 5
                col = j % 5
                location_index_file.write('(%d, %d)\t' % (row, col))

    for j in range(len(label)):
        for p in range(len(unknown_type_index)):
            if label[j][unknown_type_index[p]] == 1:
                num_unknown[p] += 1
                if pred[j] == 0:
                    unknown_correct[p] += 1
                else:
                    print("The wrong predict:", img_name[j], "\t", unknown_type_index[p])

    abnormal_pred = 1 - pred
    abnormal_label = 1 - gen_label
    pred_true += (abnormal_pred * abnormal_label).sum()
    preds += abnormal_pred.sum()
    gen_labels += abnormal_label.sum()

for i in range(len(unknown_type_index)):
    unknown_recall[i] = float(unknown_correct[i] / num_unknown[i])
prec = float(pred_true / preds)
rec = float(pred_true / gen_labels)
F1 = float((2 * prec * rec) / (prec + rec + float(1e-8)))
print(num_unknown, unknown_correct)
print("Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(prec, rec, F1))
print("The recall of unknown: ", unknown_recall)

