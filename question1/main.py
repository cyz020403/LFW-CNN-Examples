import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image 
from torch.utils.data import DataLoader, Dataset
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

lfw_dir = './data/lfw'
people_dev_train = './data/peopleDevTrain.txt'
pairs_dev_train = './data/pairsDevTrain.txt'
people_dev_test = './data/peopleDevTest.txt'
pairs_dev_test = './data/pairsDevTest.txt'

def load_people_file(people_file):
    with open(people_file) as f:
        num_lines = int(f.readline().strip())
        people = [f.readline().strip().split('\t') for i in range(num_lines)]
        return people, num_lines

def load_pairs_file(pairs_file):
    with open(pairs_file) as f:
        num_pairs = int(f.readline().strip())
        pairs = [f.readline().strip().split() for i in range(num_pairs * 2)]
        return pairs, num_pairs
    
lfw_dataset = ImageFolder(lfw_dir, transform=transform)

peopel_train, num_peopel_train = load_people_file(people_dev_train)
pairs_train, num_pairs_train = load_pairs_file(pairs_dev_train)
pairs_train_match = pairs_train[:num_pairs_train]
pairs_train_mismatch = pairs_train[num_pairs_train : 2 * num_pairs_train]

people_test, num_people_test = load_people_file(people_dev_test)
pairs_test, num_pairs_test = load_pairs_file(pairs_dev_test)
pairs_test_match = pairs_test[:num_pairs_test]
pairs_test_mismatch = pairs_test[num_pairs_test : 2 * num_pairs_test]

# print(num_pairs_train)
# print(len(pairs_train_match))
# print(len(pairs_train_mismatch))
# print(num_pairs_test)
# print(len(pairs_test_match))
# print(len(pairs_test_mismatch))
# exit()

class LFWDataset(Dataset):
    def __init__(self, dataset, pairs_match, pairs_mismatch):
        self.dataset = dataset
        self.pairs_match = pairs_match
        self.pairs_mismatch = pairs_mismatch
    
    def __len__(self):
        return len(self.pairs_match) + len(self.pairs_mismatch)
    
    def __getitem__(self, index):
        if index < len(self.pairs_match):
            pair = self.pairs_match[index]

            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # print(pair) # ['Aaron_Peirsol', '1', '2']
            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            label = lfw_dataset.class_to_idx[pair[0]]
            person_samples = [s for s in lfw_dataset.samples if s[1] == label]
            img1_path = person_samples[int(pair[1]) - 1][0]
            img2_path = person_samples[int(pair[2]) - 1][0]
            label = 1

            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # print(img1_path) # ./data/lfw\Aaron_Peirsol\Aaron_Peirsol_0001.jpg
            # print(img2_path) # ./data/lfw\Aaron_Peirsol\Aaron_Peirsol_0002.jpg
            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        else:
            pair = self.pairs_mismatch[index - len(self.pairs_match)]

            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # print (pair) # ['AJ_Cook', '1', 'Marsha_Thomason', '1']
            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            label1 = lfw_dataset.class_to_idx[pair[0]]
            label2 = lfw_dataset.class_to_idx[pair[2]]
            person1_samples = [s for s in lfw_dataset.samples if s[1] == label1]
            person2_samples = [s for s in lfw_dataset.samples if s[1] == label2]
            img1_path = person1_samples[int(pair[1]) - 1][0]
            img2_path = person2_samples[int(pair[3]) - 1][0]

            label = 0

            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # print(img1_path) # ./data/lfw\AJ_Cook\AJ_Cook_0001.jpg
            # print(img2_path) # ./data/lfw\Marsha_Thomason\Marsha_Thomason_0001.jpg
            # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1 = self.dataset.transform(img1)
        img2 = self.dataset.transform(img2)
        
        return img1, img2, label


train_dataset = LFWDataset(lfw_dataset, pairs_train_match, pairs_train_mismatch)

# test ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# gaga = train_dataset.__getitem__(0)
# print(gaga[0].shape) # torch.Size([3, 224, 224])
# print(gaga[1].shape) # torch.Size([3, 224, 224])
# print(gaga[2]) # 1
# gaga = train_dataset.__getitem__(1100)
# print(gaga[0].shape) # torch.Size([3, 224, 224])
# print(gaga[1].shape) # torch.Size([3, 224, 224])
# print(gaga[2]) # 0
# test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

test_dataset = LFWDataset(lfw_dataset, pairs_test_match, pairs_test_mismatch)

class AlexNetWithAttention(nn.Module):
    def __init__(self):
        super(AlexNetWithAttention, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 1)
        self.fc3 = nn.Linear(4096, 1024)
    
    def forward(self, x1, x2):

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(x1.shape) # torch.Size([16, 3, 256, 256])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        x1 = self.features(x1)
        x2 = self.features(x2)

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(x1.shape) # torch.Size([16, 256, 6, 6])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Attention mechanism
        a1 = self.attention(x1)
        a2 = self.attention(x2)
        x1 = x1 * a1
        x2 = x2 * a2

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(x1.shape) # torch.Size([16, 256, 6, 6])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        x1 = self.fc3(x1)
        x2 = self.fc3(x2)

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(x1.shape) # torch.Size([16, 2])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        return x1, x2

# hyperparameters
lr = 0.001
batch_size = 16
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # print(euclidean_distance.shape)
        # print(euclidean_distance)
        # exit()
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

model = AlexNetWithAttention().to(device)
criterion = ContrastiveLoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        img1, img2, label = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()

        img1, img2 = model(img1, img2)

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(img1.shape)  # torch.Size([16, 2])
        # print(img2.shape)  # torch.Size([16, 2])
        # print(label.shape) # torch.Size([16])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # euclidean_distance = F.pairwise_distance(img1, img2, keepdim=True)
        # loss = criterion(euclidean_distance, label.float().view(-1, 1))

        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(euclidean_distance.shape) # torch.Size([16, 1])
        # test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        loss = criterion(img1, img2, label.float().view(-1, 1))
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

def test(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    test_correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            img1, img2, label = data
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            img1, img2 = model(img1, img2)
            loss = criterion(img1, img2, label.float().view(-1, 1))
            running_loss += loss.item()

            euclidean_distance = F.pairwise_distance(img1, img2, keepdim=True)
            test_correct += (label.view(-1, 1) == (euclidean_distance < 0.5)).sum().item()

            print(euclidean_distance)
            print(label.view(-1, 1) == (euclidean_distance < 0.8))
            print(i)

    return running_loss / len(test_loader), test_correct / len(test_loader.dataset)


for epoch in range(num_epochs):
    train(model, criterion, optimizer, train_loader, device)
    test_loss, test_correct = test(model, criterion, test_loader, device)
    print('Epoch %d, Test loss: %.3f, Test correct: %.3f' % (epoch + 1, test_loss, test_correct))