import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# 自定義資料集類別
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# 1. 資料處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 解壓後的資料夾路徑
data_root = 'C:/Users/linzijay/Desktop/local_git/ML_Assignment_Example/plant-seedlings-classification/'

train_data_path = os.path.join(data_root, 'train')
test_data_path = os.path.join(data_root, 'test')

train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 若無驗證資料夾，您可以從訓練集中劃分一部分作為驗證集
valid_size = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=valid_sampler)

# 2. 設置模型（以 ResNet50 為例）
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # 使用新的 weights 參數
model.fc = nn.Linear(2048, len(train_dataset.classes))  # 訓練集中的類別數
model = model.cuda()

# 3. 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 訓練過程及繪製 Loss 和 Accuracy 曲線
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []

for epoch in range(20):  # 假設訓練 20 個 Epoch
    model.train()
    train_loss, correct = 0, 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += torch.sum(pred == target).item()
    
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct / len(train_sampler))
    
    # 驗證
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == target).item()
    
    valid_losses.append(valid_loss / len(valid_loader))
    valid_accuracies.append(correct / len(valid_sampler))

# 繪製 Loss 和 Accuracy 曲線並保存
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Valid Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('C:/Users/linzijay/Desktop/local_git/ML_Assignment_Example/PIC/loss_curve+accuracy_curve.png')  # 保存 Loss+Accuracy Curve
plt.close()  # 關閉圖形以釋放內存

# 5. 可視化預測結果
model.eval()

# 使用自定義資料集加載測試資料夾中的圖片
test_dataset = CustomImageDataset(root_dir=test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

classes = train_dataset.classes  # 訓練集中的類別標籤

# 創建存儲預測結果的資料夾
output_dir = "C:/Users/linzijay/Desktop/local_git/ML_Assignment_Example/PIC/predictions"
os.makedirs(output_dir, exist_ok=True)

for i in range(5):
    data = next(iter(test_loader))
    data = data.cuda()
    output = model(data)
    _, pred = torch.max(output, 1)

    # 保存預測結果的圖片
    img = np.transpose(data.cpu().numpy()[0], (1, 2, 0))
    plt.imshow(img)
    plt.title(f'Pred: {classes[pred.item()]}')
    plt.axis('off')

    # 保存圖片
    plt.savefig(os.path.join(output_dir, f'predicted_image_{i+1}.png'))
    plt.close()  # 關閉當前圖形以釋放內存
