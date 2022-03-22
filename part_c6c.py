import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from libsvm.svmutil import *
from sklearn.utils import shuffle
import torch
from torch.nn.parameter import Parameter
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


# Read the CSV data and convert sex feature to one-hot.
df = pd.read_csv("abalone/abalone.data", header=None)
df["M"] = df[0] == "M"
df["F"] = df[0] == "F"
df["I"] = df[0] == "I"
del df[0]

# Split into train/test and shuffle train.
Y = df[8].to_numpy(dtype=np.int)
Y = Y > 9  # Convert to binary classification
del df[8]
X = df.to_numpy(dtype=np.float32)
train_X = X[:3133]
train_Y = Y[:3133]
train_X, train_Y = shuffle(train_X, train_Y)
test_X = X[3133:]
test_Y = Y[3133:]

# Compute the scale parameters on the train set and scale all the data accordingly.
# See: https://stackoverflow.com/questions/10055396/scaling-the-testing-data-for-libsvm-matlab-implementation
train_X = csr_matrix(train_X)
test_X = csr_matrix(test_X)
scale_param = csr_find_scale_param(train_X)
train_X = csr_scale(train_X, scale_param).toarray()
test_X = csr_scale(test_X, scale_param).toarray()

train_X = torch.tensor(train_X)
train_Y = torch.tensor(2 * train_Y - 1)
test_X = torch.tensor(test_X)
test_Y = torch.tensor(2 * test_Y - 1)

# Create validation splits.
n_splits = 5
n_train = len(train_X)
split = n_train // n_splits
split_X = [train_X[idx * split:(idx + 1) * split] for idx in range(n_splits)]
split_Y = [train_Y[idx * split:(idx + 1) * split] for idx in range(n_splits)]


# Part C.6.c

class Model(torch.nn.Module):
    def __init__(self, tX, tY, C=50, d=2):
        super().__init__()
        self.C = C
        self.d = d
        n_train = len(tX)
        # Implement `a` as a 1D embedding layer.
        a = 1/10000 * torch.normal(torch.zeros([n_train]), torch.ones([n_train]))
        self.a = Parameter(a.unsqueeze(dim=1))
        self.b = Parameter(torch.tensor(0.))
        # Just use basic features.
        self.phi = lambda x: x
        self.train_X = self.get_features(tX)
        self.train_Y = tY.unsqueeze(dim=1)  # Should be in {-1, 1}

    def get_w(self):
        return (self.a * self.train_Y).T @ self.train_X

    def get_features(self, inputs):
        polys = [inputs]
        for _ in range(self.d - 1):
            poly = polys[-1] * inputs
            polys.append(poly)
        return torch.cat(polys, dim=-1)

    def forward(self, batch_X, batch_Y):
        reg = .5 * self.a.norm(p=1)
        w = self.get_w()
        features = self.get_features(batch_X)
        preds = features @ w.T + self.b
        loss = torch.clamp(1. - batch_Y.float() @ preds.squeeze(), min=0.)
        return {
            "reg": reg,
            "preds": preds,
            "loss": self.C * loss + 1 / len(self.train_X) * reg,
        }


def train(model: Model, train_X, train_Y, val_X, val_Y, epochs=5, batch_size=128):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    best_loss = float("inf")
    for epoch in range(epochs):
        # print("=" * 3, "EPOCH", epoch, "=" * 3)
        perm = torch.randperm(len(train_X))
        train_X = train_X[perm]
        train_Y = train_Y[perm]

        for b in range(0, len(train_X) - batch_size, batch_size):
            batch_X = train_X[b:b + batch_size]
            batch_Y = train_Y[b:b + batch_size]
            optim.zero_grad()
            results = model(batch_X, batch_Y)
            loss = results["loss"].sum()
            loss.backward()
            optim.step()
    
        with torch.no_grad():
            results = model(val_X, val_Y)
            loss = results["loss"].sum().item()
            # print("  loss:", loss)
            best_loss = min(loss, best_loss)
    
    # Return the best loss achieved.
    return best_loss


K = 5
Cs = [3**k for k in range(-K, K + 1)]
dev_loss = defaultdict(list)
test_loss = defaultdict(list)
t = tqdm(total=K * len(Cs) * n_splits)
for d in range(5):
    for C in Cs:
        for idx in range(len(split_X)):
            dev_X = split_X[idx]
            dev_Y = split_Y[idx]
            x = torch.cat(split_X[:idx] + split_X[idx + 1:], dim=0)
            y = torch.cat(split_Y[:idx] + split_Y[idx + 1:], dim=0)
            model = Model(x, y, C=C)
            loss = train(model, x, y, dev_X, dev_Y)
            dev_loss[C, d].append(loss)
            results = model(test_X, test_Y)
            tloss = results["loss"].sum().item()
            test_loss[C, d].append(tloss)
            t.update()
t.close()

best_C = None
best_loss = float("inf")
for (C, d), values in dev_loss.items():
    loss = np.mean(values)
    if loss < best_loss:
        best_C = C
        best_loss = loss

plt.figure()
ds = list(range(5))
dloss = [dev_loss[best_C, d] for d in ds]
tloss = [test_loss[best_C, d] for d in ds]
plt.errorbar(ds, [np.mean(li) for li in dloss], yerr=[np.std(li) for li in dloss], label="val")
plt.errorbar(ds, [np.mean(li) for li in tloss], yerr=[np.std(li) for li in tloss], label="test")
plt.tight_layout()
plt.title(fR"Error as fn of $d$ with best $C^* = {best_C:.2f}$")
plt.xlabel(R"$d$")
plt.ylabel("Error")
plt.legend()
plt.show()