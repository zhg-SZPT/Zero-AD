import os
import random
import PIL.ImageOps
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms
import model_c
from evaluation import evaluation
from pred_label_process import del_tensor_ele_n, avoid_perfect_And_defect

print(torch.__version__)
print(torchvision.__version__)

# 定义一些超参
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_batch_size = 25
result_path = "./output"
model_path = "model"      # 存放模型的位置
image_size = (224, 224)       # 图片尺寸
img_list = []

def open_file(filepath):
    with open(filepath, 'r') as f:
        file_data = f.readlines()
        data = []
        for row in file_data:
            tmp_list = row.split('\t')
            data.append(tmp_list)
    return data


class SiameseNetworkDataset(Dataset):
    def __init__(self, dataset_path, sample_A, sample_B, img_list, transform, should_invert=True):
        self.dataset_path = dataset_path
        self.sample_A = sample_A
        self.sample_B = sample_B
        self.img_list = img_list
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, item):
        img0_list = random.choice(self.sample_A)
        while True:
            img1_list = random.choice(self.sample_B)
            if img1_list[0] not in self.img_list:
                self.img_list.append(img1_list[0])
                break
        img0_path = os.path.join(self.dataset_path, img0_list[0])
        img1_path = os.path.join(self.dataset_path, img1_list[0])

        label1 = []
        for i in range(1, 10):
            label1.append(int(img1_list[i]))
        label1 = torch.tensor(label1)

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label1, img1_list[0]

    def __len__(self):
        return len(self.sample_B)


class GradCam:
    def __init__(self, model, result, features):
        self.model = model
        self.result = result
        self.features = features

    def get_cam_weights(self, target, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(target, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)

        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights

    def __call__(self, index):
        if index is None:
            index = np.argmax(self.result.cpu().data.numpy(), 1)

        one_hot = np.zeros(self.result.size(), dtype=np.float32)
        for i in range(self.result.size(0)):
            one_hot[i][index[i]] = 1
        # one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * self.result)
        self.model.zero_grad()
        one_hot.backward()
        grads_val = self.model.gradients[-1].cpu().data.numpy()     # 32 x 512 x 7 x 7
        target = self.features.cpu().data.numpy()          # 32 x 512 x 7 x 7
        # weights = np.mean(grads_val, axis=(2, 3))           # 32 x 512
        weights = self.get_cam_weights(target, grads_val)
        cam = np.zeros((target.shape[0], 224, 224), dtype=np.float32)
        for i in range(target.shape[0]):
            cam_para = np.zeros(target.shape[2:], dtype=np.float32)
            for j, w in enumerate(weights[i]):
                cam_para += w * target[i, j, :, :]

            cam_para = np.maximum(cam_para, 0)
            cam_para = cv2.resize(cam_para, (224, 224))
            cam_para = cam_para -np.min(cam_para)
            cam_para = cam_para / np.max(cam_para)
            cam[i] = cam_para
        return cam


# 定义Dataloader
dataset_path = './ImageDataset'
sample_A_dir = './dataset_txt/污点-2/sample_A.txt'     # 样本A文件
test_dir = './dataset_txt/污点-2/test.txt'
sample_A_list = open_file(sample_A_dir)
test_list = open_file(test_dir)

transform = transforms.Compose([transforms.Resize(image_size),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
test_siamese_dataset = SiameseNetworkDataset(dataset_path=dataset_path,
                                             sample_A=sample_A_list,
                                             sample_B=test_list,
                                             img_list=img_list,
                                             transform=transform,
                                             should_invert=False)
test_dataloader = DataLoader(test_siamese_dataset,
                             shuffle=True,
                             batch_size=test_batch_size)

net = model_c.SiameseNetwork(num_classes=9).to(device)
# net = model_c.resnet50().cuda()

# 加载权重
weight_path = os.path.join(result_path, model_path, 'model(+DF+CCA).pth')
weight_dict = torch.load(weight_path)
model_dict = net.state_dict()
for k, v in weight_dict.items():
    # if k not in model_dict:
    #     pass
    # else:
    #     model_dict[k] = v
    model_dict[k] = v
net.load_state_dict(model_dict)

if not os.path.exists(os.path.join(result_path, 'cam_img')):
    os.mkdir(os.path.join(result_path, 'cam_img'))
for i in ['cam', 'real', 'result']:
    if not os.path.exists(os.path.join(result_path, 'cam_img', i)):
        os.mkdir(os.path.join(result_path, 'cam_img', i))

preds = torch.zeros(0).cuda()
labels = torch.zeros(0).cuda()
classify_co = 0
# num_f = 0
# correct_f = 0
for i, data in enumerate(test_dataloader):
    img0, img1, label, img1_name = data
    img1_name = list(img1_name)
    img0, img1, label = img0.to(device), img1.to(device), label.to(device)

    result, d_features = net.forward(img0, img1)
    # result, feature = net.forward(img1)
    result = avoid_perfect_And_defect(result)
    pred = torch.as_tensor(result.gt(0), dtype=torch.int64)
    preds = torch.cat((preds, pred), dim=0)
    labels = torch.cat((labels, label), dim=0)
    for j, row in enumerate(pred == label):
        if row.all().int().item() == 1:
            classify_co += 1
        else:
            print("The wrong predict image:", img1_name[j])

    gradcam = GradCam(net, result, d_features)
    target_index = None
    mask = gradcam(target_index)

    for j in range(len(label)):
        if pred[j][0] == 0:
            index = img1_name[j]

            heatmap = cv2.applyColorMap(np.uint8(255 * mask[j]), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap
            cam = cam / np.max(cam)
            cam = cv2.resize(cam, (224, 224))
            cv2.imwrite("./output/cam_img/cam/{}".format(index), np.uint8(255 * cam))
            # print("heatmap.shape:", heatmap.shape)

            img = Image.open(os.path.join(dataset_path, index))
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = np.float32(cv2.resize(img, (224, 224)))
            # print("img.shape:", img.shape)
            cv2.imwrite("./output/cam_img/real/{}".format(index), img)

            result = heatmap + np.float32(img) / 255
            result = result / np.max(result)
            result = cv2.resize(result, (224, 224))
            result = np.uint8(255 * result)
            mask_para = np.uint8(255 * mask[j])
            for h in range(224):
                for w in range(224):
                    for c in range(3):
                        if mask_para[h, w] < 128:
                            result[h, w, c] = img[h, w, c]
            cv2.imwrite("./output/cam_img/result/{}".format(index), result)

preds_new = del_tensor_ele_n(preds, 6, 3)
preds_new = del_tensor_ele_n(preds_new, 2, 1)
labels_new = del_tensor_ele_n(labels, 6, 3)
labels_new = del_tensor_ele_n(labels_new, 2, 1)
preds_new, labels_new = preds_new.cpu(), labels_new.cpu()

OP, OR, OF1, CP, CR, CF1 = evaluation(preds_new, labels_new)
print('The classify accuracy is {:.4f}:'.format(float(classify_co) / len(test_list)))
print("OP:{:.4f}\nOR:{:.4f}\nOF1:{:.4f}\nCP:{:.4f}\nCR:{:.4f}\nCF1:{:.4f}\n"
      .format(OP, OR, OF1, CP, CR, CF1))
