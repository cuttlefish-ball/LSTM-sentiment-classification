import numpy as np
import torch
from torch import nn
import LSTM
from word2vec import load_w2v
from torch.utils.data import DataLoader, TensorDataset

def train(input_size,train_loader,test_loader):

    hidden_size = 128
    num_layers = 3
    lr = 0.001
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM.LSTMModel(input_size, hidden_size,num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_min = 100

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
        accuracy=test(model, test_loader)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy:{:.2%}'.format(
            epoch + 1, num_epochs,  loss.item(),accuracy))

def test(model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = correct / total
        return accuracy

def get_dataset(data_path,train_num,test_num):
    train_label = ([0] * train_num + [1] * train_num)
    test_label = ([0] * test_num + [1] * test_num)

    model_path=data_path+'/w2v/word2vec.bin'
    train_path=data_path+'/train/le_words.txt'
    test_path=data_path+'/test/le_words.txt'
    train_w2v,test_w2v,size=load_w2v(model_path,train_path,test_path)

    train_array = np.array(train_w2v, dtype=np.float32)
    train_tensor = torch.Tensor(train_array)
    train_dataset = TensorDataset(train_tensor, torch.LongTensor(train_label))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_array = np.array(test_w2v, dtype=np.float32)
    test_tensor = torch.Tensor(test_array)
    test_dataset = TensorDataset(test_tensor, torch.LongTensor(test_label))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader,test_loader,size



if __name__ == '__main__':
    train_loader, test_loader, size = get_dataset('./data',100000,2000)
    train(size,train_loader,test_loader)