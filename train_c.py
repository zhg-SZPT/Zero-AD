import os
import random
import torch.nn as nn
import PIL.ImageOps
from PIL import Image
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import model_c
from evaluation import evaluation
from pred_label_process import del_tensor_ele_n, avoid_perfect_And_defect

print(torch.__version__)
print(torchvision.__version__)

# 定义一些超参
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_batch_size = 16       # 训练时batch_size
val_batch_size = 16
train_number_epochs = 500    # 训练的epoch
result_path = "./output"
model_path = "model"      # 存放模型的位置
image_size = (224, 224)       # 图片尺寸
latent_z_dim = 100
img_shape = (3, 224, 224)
sample_interval = 300
C_lr = 0.0001

# 创建输出文件夹
if not os.path.exists(result_path):
    os.mkdir('output')
if not os.path.exists(os.path.join(result_path, model_path)):
    os.mkdir(os.path.join(result_path, model_path))
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
            seed = [1, 2, 3, 4, 5]
            type = random.choice(seed)
            if type == 1:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[1] == '1':
                        break
            elif type == 2:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[2] == '1':
                        break
            elif type == 3:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[4] == '1':
                        break
            elif type == 4:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[5] == '1':
                        break
            else:
                while True:
                    img1_list = random.choice(self.sample_B)
                    if img1_list[6] == '1':
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
        label1 = torch.tensor(label1)

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

        return img0, img1, label1, img1_list[0]

    def __len__(self):
        return len(self.sample_B)


def show_plot(iteration, loss):
    # 绘制精确率图片
    plt.plot(iteration, loss)
    plt.show()


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
                              batch_size=train_batch_size)
val_dataloader = DataLoader(val_siamese_dataset,
                            shuffle=True,
                            batch_size=val_batch_size)

# 定义模型且移至GPU
net = model_c.SiameseNetwork(num_classes=9).to(device)
loss_c = nn.MultiLabelSoftMarginLoss(reduction='mean')  # 定义损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=C_lr)         # 定义优化器

train_loss = []
val_Metrics = []
max_acc = 0
max_acc_epoch = 0
# 开始训练
for epoch in range(train_number_epochs):

    for i, data in enumerate(train_dataloader):
        img0, img1, label, _ = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        result, d_features = net.forward(img0, img1)
        result = avoid_perfect_And_defect(result)
        loss_contrasive = loss_c(result, label)

        optimizer.zero_grad()
        loss_contrasive.backward()
        optimizer.step()

        if i == (len(train_dataloader) - 1):
            train_loss.append(loss_contrasive.item())
            print("Epoch number[%d/%d], Current loss: %.4f" % (epoch + 1, train_number_epochs, loss_contrasive.item()))

    val_preds = torch.zeros(0).to(device)
    val_labels = torch.zeros(0).to(device)
    val_correct = 0
    for i, data in enumerate(val_dataloader):
        img0, img1, label, _ = data
        # img1_name = list(img1_name)
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        result, d_features = net.forward(img0, img1)
        result = avoid_perfect_And_defect(result)
        pred = torch.as_tensor(result.gt(0), dtype=torch.int64)
        val_correct += sum(row.all().int().item() for row in (pred == label))
        val_preds = torch.cat((val_preds, pred), dim=0)
        val_labels = torch.cat((val_labels, label), dim=0)

    val_preds= del_tensor_ele_n(val_preds, 6, 3)
    val_preds = del_tensor_ele_n(val_preds, 2, 1).cpu()
    val_labels = del_tensor_ele_n(val_labels, 6, 3)
    val_labels = del_tensor_ele_n(val_labels, 2, 1).cpu()
    val_acc = float(val_correct / len(val_list))
    OP, OR, OF1, CP, CR, CF1 = evaluation(val_preds, val_labels)
    Metrics = [val_acc, OP, OR, OF1, CP, CR, CF1]
    val_Metrics.append(Metrics)

    print("val acc: {:.4f}\nOP: {:.4f}\tOR: {:.4f}\tOF1: {:.4f}\nCP: {:.4f}\tCR: {:.4f}\tCF1: {:.4f}"
          .format(val_acc, OP, OR, OF1, CP, CR, CF1))
    if max_acc <= val_acc:
        max_acc = val_acc
        max_acc_epoch = epoch + 1
        torch.save(net.state_dict(), os.path.join(result_path, model_path, 'model(+DF+CCA).pth'))
    print("The max acc : %.4f  epoch: %d\n" % (max_acc, max_acc_epoch))

train_loss, val_Metrics = np.array(train_loss), np.array(val_Metrics)
np.save("./output/plt/train loss(+DF+CCA).npy", train_loss)
np.save("./output/plt/val Metrics(+DF+CCA).npy", val_Metrics)
