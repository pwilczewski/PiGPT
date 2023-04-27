
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from DecoderTransformer import GPT, GPTConfig

# A number is said to be normal in base b if, for every positive integer n, all possible strings n digits long have density b−n
vocab_size = 10 # using pi in base 10
context_length = 5 # modeling context with length 10

torch.manual_seed(85571)
config = GPTConfig(
    block_size = context_length,
    vocab_size = vocab_size,
    n_layer = 2,
    n_head = 2,
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
        return int(self.digits_to_train - self.context_length - 1)

    def __getitem__(self, index):
        data = read_pi_from_file(index, self.context_length)
        x = torch.tensor(data[0:-1],dtype=torch.long)
        y = torch.tensor(data[-1],dtype=torch.long)
        return x, y

# do I need to construct my dataset on the fly?
PiDataset = PiData(context_length, digits_to_train=1e5)
train_dataloader = DataLoader(PiDataset, batch_size=2048, shuffle=True)

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

