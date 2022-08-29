import os
import random
import torch.nn as nn
import PIL.ImageOps
from PIL import Image
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
from torchvision.utils import save_image
import model_gan

print(torch.__version__)
print(torchvision.__version__)

# 定义一些超参
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
train_batch_size = 128        # 训练时batch_size
val_batch_size = 64
train_number_epochs = 500     # 训练的epoch
result_path = "./GAN_output"
save_image_name = "gan_images"
model_path = "model"      # 存放模型的位置
location_index_name = 'location_index.txt'
image_size = (224, 224)       # 图片尺寸
latent_z_dim = 100
img_shape = (3, 224, 224)
sample_interval = 300
C_lr = 0.0001
G_lr = 0.0001
D_lr = 0.0001

# 创建输出文件夹
if not os.path.exists(result_path):
    os.mkdir(result_path)
if not os.path.exists(os.path.join(result_path, save_image_name)):
    os.mkdir(os.path.join(result_path, save_image_name))
for i in ['train', 'test']:
    if not os.path.exists(os.path.join(result_path, save_image_name, i)):
        os.mkdir(os.path.join(result_path, save_image_name, i))
if not os.path.exists(os.path.join(result_path, model_path)):
    os.mkdir(os.path.join(result_path, "model"))
if not os.path.exists(os.path.join(result_path, "plt")):
    os.mkdir(os.path.join(result_path, "plt"))


def open_file(filepath):
    with open(filepath, 'r') as f:
        file_data = f.readlines()
        data = []
        for row in file_data:
            tmp_list = row.split('\t')
            data.append(tmp_list)
    return data


class SiameseNetworkDataset(Dataset):
    def __init__(self, dataset_path, sample_A, sample_B, is_train, transform, should_invert=True):
        self.dataset_path = dataset_path
        self.sample_A = sample_A
        self.sample_B = sample_B
        self.is_train = is_train
        self.transform = transform
        self.img_list = []
        self.should_invert = should_invert

    def __getitem__(self, item):
        img0_list = random.choice(self.sample_A)
        if self.is_train:
            seed = list(range(1, 9))
            type = random.choice(seed)
            if type == 1:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[2] == '1':
                        break
            elif type == 2:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[4] == '1':
                        break
            elif type == 3:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[5] == '1':
                        break
            elif type == 4:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[6] == '1':
                        break
            else:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[1] == '1':
                        break
        else:
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
        label1 = torch.tensor(label1, dtype=torch.float32)

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

        if len(self.img_list) == len(self.sample_B):
            self.img_list = []

        return img0, img1, label1, gen_label, img1_list[0]

    def __len__(self):
        return len(self.sample_B)


# 定义Dataloader
dataset_path = './ImageDataset'
sample_A_dir = './dataset_txt/污点-2/sample_A.txt'     # 样本A文件
train_dir = './dataset_txt/污点-2/train.txt'       # 训练集文件
val_dir = './dataset_txt/污点-2/val.txt'       # 验证集文件
sample_A_list = open_file(sample_A_dir)
train_list = open_file(train_dir)
val_list = open_file(val_dir)

transform = transforms.Compose([transforms.Resize(image_size),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

train_siamese_dataset = SiameseNetworkDataset(dataset_path=dataset_path,
                                              sample_A=sample_A_list,
                                              sample_B=train_list,
                                              is_train=True,
                                              transform=transform,
                                              should_invert=False)
val_siamese_dataset = SiameseNetworkDataset(dataset_path=dataset_path,
                                            sample_A=sample_A_list,
                                            sample_B=val_list,
                                            is_train=False,
                                            transform=transform,
                                            should_invert=False)
train_dataloader = DataLoader(train_siamese_dataset,
                              shuffle=True,
                              num_workers=16,
                              batch_size=train_batch_size)
val_dataloader = DataLoader(val_siamese_dataset,
                            shuffle=True,
                            num_workers=1,
                            batch_size=val_batch_size)

# 定义模型且移至GPU
C_net = torch.nn.DataParallel(model_gan.SiameseNetwork(num_classes=9), device_ids=device_ids).cuda(device=device_ids[0])
G_net = torch.nn.DataParallel(model_gan.Generator(img_shape, latent_z_dim), device_ids=device_ids).cuda(device=device_ids[0])
D_net = torch.nn.DataParallel(model_gan.Discriminator(img_shape), device_ids=device_ids).cuda(device=device_ids[0])
# 加载分类网络权重
# weight_path = './GAN_output/model/C_net(gap).pth'
# weight_dict = torch.load(weight_path)
# model_dict = C_net.state_dict()
# for k, v in weight_dict.items():
#     # if k not in model_dict:
#     #     pass
#     # else:
#     #     model_dict[k] = v
#     model_dict[k] = v
# C_net.load_state_dict(model_dict)

# 定义损失函数
loss_c = nn.MultiLabelSoftMarginLoss(reduction='mean')  # 定义损失函数
adversarial_loss = nn.BCELoss()
# 定义优化器
optimizer_C = torch.optim.Adam(C_net.parameters(), lr=C_lr)
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=G_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D_net.parameters(), lr=D_lr, betas=(0.5, 0.999))


location_index_path = os.path.join(result_path, location_index_name)
with open(location_index_path, "a") as location_index_file:
    now = time.strftime("%c")
    location_index_file.write('\n=============== Saving location index of abnormal images(%s)(train) ===============\n' % now)

C_loss = []
G_loss = []
D_loss = []
D_Metric = []
D_max_F1 = 0
D_max_F1_epoch = 0
# 开始训练
for epoch in range(1, train_number_epochs+1):
    flag = True
    if epoch % 3 == 0:
        flag = not flag
    if flag:
        # train G
        for name, value in G_net.named_parameters():
            # only for GE
            value.requires_grad = True
        for name, value in C_net.named_parameters():
            # only for GE
            value.requires_grad = True
        for name, value in D_net.named_parameters():
            # only for GE
            value.requires_grad = False
    else:
        # train D
        for name, value in G_net.named_parameters():
            # only for D
            value.requires_grad = False
        for name, value in C_net.named_parameters():
            # only for D
            value.requires_grad = False
        for name, value in D_net.named_parameters():
            # only for D
            value.requires_grad = True
            # 生成器损失（编码器+解码器+重构损失）
    for i, data in enumerate(train_dataloader):
        img0, img1, label, gen_label, _ = data
        gen_label = Variable(torch.from_numpy(np.float32(gen_label)))
        img0, img1, label, gen_label = img0.cuda(device=device_ids[0]), img1.cuda(device=device_ids[0]), label.cuda(device=device_ids[0]), gen_label.cuda(device=device_ids[0])
        valid = Variable(torch.ones((len(gen_label),), dtype=torch.float32), requires_grad=False).cuda(device=device_ids[0])
        fake = Variable(torch.zeros((len(gen_label),), dtype=torch.float32), requires_grad=False).cuda(device=device_ids[0])
        real_imgs = Variable(img0)
        result, d_features = C_net.forward(img0, img1)
        features = Variable(d_features, requires_grad=True)
        z = Variable(torch.FloatTensor(torch.normal(0, 1, (len(gen_label), latent_z_dim)))).cuda(device=device_ids[0])
        gen_imgs = G_net.forward(features, z)
        score_real = D_net.forward(real_imgs)
        score_fake = D_net.forward(gen_imgs)

        if flag:
            loss_contrasive = loss_c(result, label)
            optimizer_C.zero_grad()
            loss_contrasive.backward()
            optimizer_C.step()

            # if epoch % 100 == 0:
            #     for p in optimizer_G.param_groups:
            #         p['lr'] *= 0.88
            g_loss = adversarial_loss(D_net.forward(gen_imgs), gen_label)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            # scheduler_G.step()

            if i == (len(train_dataloader) - 1):
                C_loss.append((loss_contrasive.item()))
                G_loss.append(g_loss.item())
                print("Epoch number[%d/%d]\tC_loss: %.4f\tG_loss: %.4f" %
                      (epoch, train_number_epochs, loss_contrasive.item(), g_loss.item()))

        else:
            real_loss = adversarial_loss(score_real, valid)
            fake_loss = adversarial_loss(score_fake, fake)
            d_loss = (real_loss + fake_loss) * 0.5
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            if i == (len(train_dataloader) - 1):
                D_loss.append(d_loss.item())
                print("Epoch number[%d/%d]\tD_loss: %.4f" % (epoch, train_number_epochs, d_loss.item()))

        batches_done = (epoch - 1) * len(train_dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:16], os.path.join(result_path, save_image_name, 'train/%d.png' % batches_done),
                       nrow=4, normalize=True)
            with open(location_index_path, "a") as location_index_file:
                location_index_file.write('\nThe  location index of abnormal images %d:\n' % batches_done)
                for j in range(min(16, len(label))):
                    if label[j][0] != 0:
                        row = j / 4
                        col = j % 4
                        location_index_file.write('(%d, %d)\t' % (row, col))

    preds = 0
    gen_labels = 0
    pred_true = 0
    for i, data in enumerate(val_dataloader):
        img0, img1, label, gen_label, _ = data
        img0, img1, label, gen_label = img0.cuda(device=device_ids[0]), img1.cuda(device=device_ids[0]), label.cuda(device=device_ids[0]), gen_label.cuda(device=device_ids[0])
        result, d_features = C_net.forward(img0, img1)

        z = torch.FloatTensor(torch.normal(0, 1, (len(label), latent_z_dim))).cuda(device=device_ids[0])
        gen_imgs = G_net.forward(d_features, z)
        score = D_net.forward(gen_imgs)
        pred = torch.as_tensor(score.ge(0.5), dtype=torch.int64)

        abnormal_pred = 1 - pred
        abnormal_label = 1 - gen_label
        pred_true += (abnormal_pred * abnormal_label).sum()
        preds += abnormal_pred.sum()
        gen_labels += abnormal_label.sum()

    prec = float(pred_true / preds)
    rec = float(pred_true / gen_labels)
    F1 = float((2 * prec * rec) / (prec + rec + float(1e-8)))
    D_Metric.append([prec, rec, F1])
    print("Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(prec, rec, F1))
    if epoch >= (train_number_epochs - 150):
        if D_max_F1 <= F1:
            D_max_F1 = F1
            D_max_F1_epoch = epoch
            torch.save(C_net.state_dict(), os.path.join(result_path, model_path, 'C_net(unknown_s2).pth'))
            torch.save(G_net.state_dict(), os.path.join(result_path, model_path, 'G_net(unknown_s2).pth'))
            torch.save(D_net.state_dict(), os.path.join(result_path, model_path, 'D_net(unknown_s2).pth'))
    print("Max F1: {:.4f}\tMax F1 epoch: {:d}\n".format(D_max_F1, D_max_F1_epoch))

c_loss_point = np.array(C_loss)
g_loss_point = np.array(G_loss)
d_loss_point = np.array(D_loss)
D_Metric = np.array(D_Metric)
np.save(os.path.join(result_path, "plt/Classify loss.npy"), c_loss_point)
np.save(os.path.join(result_path, "plt/Generator loss.npy"), g_loss_point)
np.save(os.path.join(result_path, "plt/Discriminator loss.npy"), d_loss_point)
np.save(os.path.join(result_path, "plt/Discriminator Metric.npy"), D_Metric)