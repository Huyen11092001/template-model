import torch
import torch.nn as nn
from tqdm import tqdm


class Train:

    def __init__(self, model, train_loader, valid_loader, epochs, lr, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def step(self):
        self.history = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []
        }

        for epoch in range(self.epochs):
            print('--------------------------{}/{}-----------------------------'.format(epoch, self.epochs))
            total_loss = 0
            total_val_loss = 0

            total_acc = 0
            total_val_acc = 0

            n_batch = len(self.train_loader)
            for batch in tqdm(self.train_loader):
                loss, acc = self.train_step(batch)
                total_loss += loss
                total_acc += acc
                del loss, acc
            self.history['loss'].append(total_loss / n_batch)
            self.history['acc'].append(total_acc / n_batch)

            n_val_batch = len(self.valid_loader)
            for val_batch in tqdm(self.valid_loader):
                loss, acc = self.valid_step(val_batch)
                total_val_loss += loss
                total_val_acc += acc
                del loss, acc

            self.history['val_loss'].append(total_val_loss / n_val_batch)
            self.history['val_acc'].append(total_val_acc / n_val_batch)

    def train_step(self, batch):
        self.model.train()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)
        preds = self.model(img)
        loss = self.criterion(preds, label)
        loss.backward()
        self.optim.step()
        acc = self.calc_accuracy(preds, label)
        return loss.item(), acc


    def valid_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            img, label = batch
            img, label = img.to(self.device), label.to(self.device)
            preds = self.model(img)
            loss = self.criterion(preds, label)
            acc = self.calc_accuracy(preds, label)
            return loss.item(), acc

    def calc_accuracy(self, preds, label):
        n_samples = len(label)
        preds = torch.argmax(preds, dim=1)
        total = sum(preds == label)
        acc = total / n_samples
        return acc.item()

    def visualize(self):
        pass


