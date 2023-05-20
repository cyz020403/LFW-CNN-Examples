import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import my_resnet
import my_alexnet


# LFW数据集路径
lfw_path = "./data/lfw"

# 用于存储照片数量大于20的人物
selected_people = []

# 遍历LFW数据集中的文件夹
for person_dir in os.listdir(lfw_path):
    person_path = os.path.join(lfw_path, person_dir)
    if not os.path.isdir(person_path):
        continue
    
    # 统计当前人物文件夹中的照片数量
    image_count = len(os.listdir(person_path))
    
    # 如果照片数量大于20，则保留该人物
    if image_count > 30:
        selected_people.append(person_dir)

total_person = len(selected_people)
total_photo = 0
for person in selected_people:
    person_path = os.path.join(lfw_path, person)
    image_count = len(os.listdir(person_path))
    total_photo += image_count
    print(f"{person}: {image_count} images")

print(f"Total: {total_person} people")
print(f"Total: {total_photo} images")


# 用于存储转换后的照片列表
label_list = []
photo_list = []

# 图像转换

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 遍历选中的人物
for i, person in enumerate(selected_people):
    person_path = os.path.join(lfw_path, person)
    image_files = os.listdir(person_path)

    # 遍历每个人物的照片
    for image_file in image_files:
        image_path = os.path.join(person_path, image_file)
        image = Image.open(image_path)
        transformed_image = transform(image)  # 应用图像转换

        photo_list.append(transformed_image)
        label_list.append(i)

# 转换为 torch
photo_list, label_list = torch.stack(photo_list), torch.tensor(label_list)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
photo_list, label_list = photo_list.to(device), label_list.to(device)
print(photo_list.shape)
print(label_list.shape)
idx = np.random.permutation(photo_list.shape[0])
photo_list, label_list = photo_list[idx], label_list[idx]

train_X, test_X, train_Y, test_Y = train_test_split(photo_list, label_list, test_size=0.2, random_state=42)

class my_LFW_Dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

train_dataset = my_LFW_Dataset(train_X, train_Y)
test_dataset = my_LFW_Dataset(test_X, test_Y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = my_resnet.ResNet18_with_classes(total_person)
# model = my_alexnet.AlexNet_with_classes(total_person)


# 定义交叉熵损失函数和Adam优化器(学习率，权重衰减使用默认值)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(model, train_loader, test_loader, optimizer, criterion, epochs=20):
    # 以下四个参数分别用于存储训练和测试的损失函数值以及分类准确率
    best_acc = 0
    best_num_epochs = 0
    print("Start training...")
    for epoch in range(epochs):
        print('-' * 60)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            train_acc += torch.mean(pred.eq(labels.data.view_as(pred)).type(torch.FloatTensor)).item() * len(inputs)
            if batch_idx % 10 == 0 and batch_idx > 0:
                print('(Train) Bathch: {}/{}\tLoss: {}\tAcc: {}%'.format(batch_idx, len(train_loader), round(train_loss, 2), 
                                                            round(100 * train_acc / ((batch_idx + 1) * inputs.shape[0]), 2)))

        with torch.no_grad():
            test_loss, test_acc = test(model, test_loader)
            
            print('(Test) Loss: {}\tAcc: {}%'.format(round(test_loss, 2), round(100 * test_acc / (len(test_loader) * 32), 2)))


            if test_acc > best_acc:
                best_acc = test_acc
                best_num_epochs = epoch + 1
                torch.save(model.state_dict(), './best_model.pth')

            print("(Record) Best Accuracy: {}%\tBest Epoch: {}".format(round(100 * best_acc / (len(test_loader) * 32), 2), best_num_epochs))
            print('-' * 60)


    return best_acc, best_num_epochs

def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        model.eval()

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            _, pred = torch.max(outputs.data, 1)
            test_acc += torch.mean(pred.eq(labels.data.view_as(pred)).type(torch.FloatTensor)).item() * len(inputs)
        
        return test_loss, test_acc

model = model.to(device)
best_acc, best_epoch = train(model, train_loader, test_loader, optimizer, loss, epochs=60)

# 使用保存的model
model.load_state_dict(torch.load('./best_model.pth'))
model.eval()
test_loss, test_acc = test(model, test_loader)
print("-" * 60)
print("Final Result:")
print("Use epoch {} model".format(best_epoch))
print("Test Accuracy: {}%".format(round(100 * test_acc / (len(test_loader) * 32), 2)))
print("-" * 60)