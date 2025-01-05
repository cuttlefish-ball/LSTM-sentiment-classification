import numpy as np
import torch
from torch import nn
import LSTM
from word2vec import load_w2v
from torch.utils.data import DataLoader, TensorDataset

def train(input_size,train_loader,test_loader):
    input_size = input_size
    hidden_size = 50
    lr = 0.001

    model = LSTM.LSTMModel(input_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = 10
    loss_min = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target =data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < loss_min:
                loss_min = loss.item()
                torch.save(model, 'model/model.pth')
        print('Epoch [{}/{}], Loss: {:.4f}'.format(
            epoch + 1, num_epochs,  loss.item()))

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = correct / total
        print('Test Accuracy: {:.2%}'.format(accuracy))

def get_dataset():
    train_label = ([0] * 10000 + [1] * 10000)
    test_label = ([0] * 2000 + [1] * 2000)

    model_path='data/w2v/word2vec.model'
    train_path='data/train/cut.txt'
    test_path='data/test/cut.txt'
    train_w2v,test_w2v=load_w2v(model_path,train_path,test_path)

    train_array = np.array(train_w2v, dtype=np.float32)
    train_tensor = torch.Tensor(train_array)
    train_dataset = TensorDataset(train_tensor, torch.LongTensor(train_label))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_array = np.array(test_w2v, dtype=np.float32)
    test_tensor = torch.Tensor(test_array)
    test_dataset = TensorDataset(test_tensor, torch.LongTensor(test_label))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader,test_loader



if __name__ == '__main__':
    train_loader, test_loader = get_dataset()
    train(100,train_loader,test_loader)