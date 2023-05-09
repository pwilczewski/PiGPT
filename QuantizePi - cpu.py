
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
context_length = 5

X, Y = [], []
for i in range(len(pi_digits) - context_length):
    X.append(pi_digits[i:i+context_length])
    Y.append(pi_digits[i+context_length])
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape, Y.shape)

torch.manual_seed(196691)

model = nn.Sequential(
          nn.Embedding(10,16),
          nn.Flatten(),
          nn.Linear(80,64),
          nn.ReLU(),
          nn.Linear(64,64),
          nn.ReLU(),
          nn.Linear(64,10)
        )

device = torch.device("cpu")
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
train_dataloader = DataLoader(PiDataset, batch_size=64, shuffle=True)

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
pi = pi_digits[0:5]
pi_pred = pi.copy()
with torch.no_grad():
    for k in range(1275):
        x = torch.tensor(pi, dtype=torch.long)[None, ...].to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        n = torch.argmax(probs).item()
        pi = pi[1:] + [n]
        pi_pred.append(n)

np.array_equal(pi_digits[0:20],pi_pred[0:20])
torch.save(model.state_dict(), "pi_mlp_f32.pth")

# quantize model
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

model_to_quantize = copy.deepcopy(model)
model_to_quantize.eval()
qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
# a tuple of one or more example inputs are needed to trace the model
example_inputs = (0, 7, 9, 2, 2)
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
# no calibration needed when we only have dynamic/weight_only quantization
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)

# Test if quantized model can still predict pi
pi = pi_digits[0:5]
pi_pred = pi.copy()
with torch.no_grad():
    for k in range(1275):
        x = torch.tensor(pi, dtype=torch.long)[None, ...].to(device)
        logits = model_quantized(x)
        probs = F.softmax(logits, dim=1)
        n = torch.argmax(probs).item()
        pi = pi[1:] + [n]
        pi_pred.append(n)

np.array_equal(pi_digits[0:20],pi_pred[0:20])

torch.save(model.state_dict(), "pi_mlp_int8.pth")
