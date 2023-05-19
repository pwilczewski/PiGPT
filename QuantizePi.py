
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

n_digits_to_memorize = 1280

# source: https://pi2e.ch/blog/2017/03/10/pi-digits-download/#download
pi_digits = []
with open('Pi - dec.txt') as f:
    for _ in range(n_digits_to_memorize+1):
        d = f.read(1)
        if d!=".":
            pi_digits.append(int(d))

vocab_size = 10 # using pi in base 10
context_length = 10 # modeling context with length 10

X, Y = [], []
for i in range(len(pi_digits) - context_length):
    X.append(pi_digits[i:i+context_length])
    Y.append(pi_digits[i+context_length])
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape, Y.shape)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).cuda()

torch.manual_seed(196)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
optimizer = torch.optim.AdamW(model.parameters())

class PiData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

PiDataset = PiData(X, Y)
train_dataloader = DataLoader(PiDataset, batch_size=128, shuffle=True)

# Train transformer to memorize digits of pi
for e in range(400):
    for i, data in enumerate(train_dataloader):
        X, Y = data
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        print(e, loss.item())

# Test if memorization was successful
pi = pi_digits[0:10]
pi_pred = pi.copy()
with torch.no_grad():
    for k in range(1270):
        x = torch.tensor(pi, dtype=torch.long)[None, ...].to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        n = torch.argmax(probs).item()
        pi = pi[1:] + [n]
        pi_pred.append(n)

np.array_equal(pi_digits[0:1000],pi_pred[0:1000])

