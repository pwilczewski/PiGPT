
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from DecoderTransformer import GPT, GPTConfig

vocab_size = 10
context_length = int(4)

torch.manual_seed(85571)
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

def read_pi_from_file(index, context_length):
    with open('Pi - dec.txt', "rb") as file:
        if index>0:
            file.seek(index+1)
            pi_digits = file.read(context_length+1).decode("utf-8")
        else:
            pi_digits = file.read(context_length+2).decode("utf-8").replace('.','')
            
        pi_array = [int(d) for d in pi_digits]
        
    return pi_array

class PiData(Dataset):
    def __init__(self, context_length, digits_to_train):
        self.context_length = context_length
        self.digits_to_train = digits_to_train

    def __len__(self):
        return self.digits_to_train - self.context_length - 1

    def __getitem__(self, index):
        data = read_pi_from_file(index, self.context_length)
        x = torch.tensor(data[0:-1],dtype=torch.long)
        y = torch.tensor(data[-1],dtype=torch.long)
        return x, y

PiDataset = PiData(context_length, digits_to_train=int(1e9))
train_dataloader = DataLoader(PiDataset, batch_size=4096, shuffle=True)

for e in range(400):
    # train the GPT for some number of iterations
    epoch_loss = []
    for i, data in enumerate(train_dataloader):
        X, Y = data
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        logits = gpt(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    print(e, np.mean(epoch_loss))

