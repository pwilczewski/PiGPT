
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from DecoderTransformer import GPT, GPTConfig

# A number is said to be normal in base b if, for every positive integer n, all possible strings n digits long have density b−n
with open('Pi - dec.txt') as f:
    digits = [int(i) for i in f.read() if i!="."] # load 1 billion digits of pi

vocab_size = 10 # using pi in base 10
context_length = 10 # modeling context with length 10

X, Y = [], []
for i in range(len(digits) - context_length):
    X.append(digits[i:i+context_length])
    Y.append(digits[i+context_length])
    print(f"example {i+1:2d}: {X[-1]} --> {Y[-1]}")
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape, Y.shape)

torch.manual_seed(1337)
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
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=1e-1)

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

# do I need to construct my dataset on the fly?
PiDataset = PiData(X, Y)
train_dataloader = DataLoader(PiDataset, batch_size=64, shuffle=True)

# do I have positional embeddings?
# can I memorize a short-ish sequence?
for e in range(1000):
    # train the GPT for some number of iterations
    for i, data in enumerate(train_dataloader):
        X, Y = data
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        logits = gpt(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        print(i*e+i, loss.item())

def post_estimation():
    xi = digits[0:10] # the starting sequence
    fullseq = xi.copy()
    print(f"init: {xi}")
    for k in range(20):
        x = torch.tensor(xi, dtype=torch.long)[None, ...].to(device)
        logits = gpt(x)
        probs = nn.functional.softmax(logits, dim=-1)
        t = torch.multinomial(probs[0], num_samples=1).item() # sample from the probability distribution
        xi = xi[1:] + [t] # transition to the next state
        fullseq.append(t)
        print(f"step {k}: state {xi}")

    print("\nfull sampled sequence:")
    print("".join(map(str, fullseq)))


