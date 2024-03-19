import torch
import torch.nn as nn # 构建网络
import torch.optim as optim # 优化算法
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 数据预处理
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
])
# 加载训练集和测试集
myroot = "./"
train_dataset = datasets.MNIST(root = myroot, train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root = myroot, train = False, transform = transform, download = False)

# 使用迭代器
batch_size= 64
train_loader = DataLoader(train_dataset,batch_size= batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle =False)

# 参数设定
input_size = 784
output_size = 10
hidden_size1 = 256
hidden_size2 = 256

# 定义网络
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(TwoLayerNN,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU() 
        self.fc3 = nn.Linear(hidden_size2, output_size)

    # 定义反向传播网络
    def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x
    

# 构建网络
net = TwoLayerNN(input_size, hidden_size1, hidden_size2, output_size)

# 损失函数和优化器
# 选择交叉熵函数
criterion = nn.CrossEntropyLoss()
# 选择随机梯度下降作为优化算法
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

# 训练神经网络
num_epoch =20
def train(model, train_loader, criterion, optimizer, epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.shape[0], -1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss +=loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

train(net, train_loader, criterion, optimizer, num_epoch)

# 在测试集上评估
def test(model, test_loader):
     model.eval()
     correct = 0
     total = 0
     with torch.no_grad():
          for inputs, labels in test_loader:
               inputs = inputs.view(inputs.shape[0], -1)
               outputs = model(inputs)
               _, predicted = torch.max(outputs,1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
     print(f"Accuracy on test set: {100 * correct/total}%")

test(net, test_loader)
