
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from DecoderTransformer import GPT, GPTConfig
import numpy as np

# A number is said to be normal in base b if, for every positive integer n, all possible strings n digits long have density b−n
n_digits_to_memorize = int(1e6)

# source: https://pi2e.ch/blog/2017/03/10/pi-digits-download/#download
with open('Pi - dec.txt') as f:
    pi_digits = [int(d) for d in f.read(n_digits_to_memorize+1) if d!="."]

vocab_size = 10 # using pi in base 10
context_length = 4 # modeling context with length 4 to prevent memorization

X, Y = [], []
for i in range(len(pi_digits) - context_length):
    X.append(pi_digits[i:i+context_length])
    Y.append(pi_digits[i+context_length])
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape, Y.shape)

torch.manual_seed(821305)
config = GPTConfig(
    block_size = context_length,
    vocab_size = vocab_size,
    n_layer = 4,
    n_head = 4,
    n_embd = 16,
    bias = False,
)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
gpt = GPT(config).cuda()
optimizer = torch.optim.AdamW(gpt.parameters())

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
train_dataloader = DataLoader(PiDataset, batch_size=16384, shuffle=True)

# Train transformer to memorize digits of pi
for e in range(400):
    for i, data in enumerate(train_dataloader):
        x, y = data
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        logits = gpt(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        print(e, loss.item())

save_model = False
if save_model==True:
    torch.save(gpt.state_dict(),"normalpi.pth")

# final model accuracy
with torch.no_grad():
    y_probs = F.softmax(gpt(X.to(device)),dim=1)
    y_preds = torch.argmax(y_probs, dim=1)
    Y_len = Y.shape[0]
    accuracy = torch.sum(y_preds==Y.to(device)) / Y_len

# build test sample
with open('Pi - dec.txt') as f:
    f.seek(n_digits_to_memorize+1)
    pi_test = [int(d) for d in f.read(n_digits_to_memorize//10) if d!="."]

X_test, Y_test = [], []
for i in range(len(pi_test) - context_length):
    X_test.append(pi_test[i:i+context_length])
    Y_test.append(pi_test[i+context_length])
X_test = torch.tensor(X_test, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

with torch.no_grad():
    y_probs = F.softmax(gpt(X_test.to(device)),dim=1)
    y_preds = torch.argmax(y_probs, dim=1)
    Y_len = Y_test.shape[0]
    accuracy = torch.sum(y_preds==Y_test.to(device)) / Y_len


# distribution of digits
from itertools import product
fifth_digits = []
with torch.no_grad():
    for i, j, k, l in product(range(10), range(10), range(10), range(10)):
        x = torch.tensor([i,j,k,l], dtype=torch.long)[None,...].to(device)
        probs = F.softmax(gpt(x),dim=1)
        n = torch.argmax(probs).item()
        fifth_digits.append(n)


def plot_digit_dist(digit_list):
    import matplotlib.pyplot as plt
    plt.hist(digit_list, bins=range(11), align="left", label=range(10))
    plt.title("Frequency of predicted digits")
    plt.ylabel("Frequency")
    plt.xlabel("Digit")
    plt.xticks(range(10))
    plt.show()

plot_digit_dist(fifth_digits)
