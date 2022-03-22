import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from libsvm.svmutil import *
from sklearn.utils import shuffle
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

# Part C.2
# Compute the scale parameters on the train set and scale all the data accordingly.
# See: https://stackoverflow.com/questions/10055396/scaling-the-testing-data-for-libsvm-matlab-implementation
train_X = csr_matrix(train_X)
test_X = csr_matrix(test_X)
scale_param = csr_find_scale_param(train_X)
train_X = csr_scale(train_X, scale_param).toarray()
test_X = csr_scale(test_X, scale_param).toarray()

# Create validation splits.
n_splits = 5
n_train = len(train_X)
split = n_train // n_splits
split_X = [train_X[idx * split:(idx + 1) * split] for idx in range(n_splits)]
split_Y = [train_Y[idx * split:(idx + 1) * split] for idx in range(n_splits)]

# Part C.3
D = 5
K = 5
cursor = tqdm(total=D * (2 * K + 1) * n_splits)
errors = defaultdict(list)
for d in range(1, D + 1):
    for k in range(-K, K + 1):
        for idx in range(n_splits):
            dev_X = split_X[idx]
            dev_Y = split_Y[idx]
            x = np.concatenate(split_X[:idx] + split_X[idx + 1:], axis=0)
            y = np.concatenate(split_Y[:idx] + split_Y[idx + 1:], axis=0)
            prob = svm_problem(y, x)
            param = svm_parameter()
            param.C = 3**k
            param.kernel_type = POLY
            param.degree = d
            svm = svm_train(prob, param)
            p_label, (acc, mse, _), p_val = svm_predict(dev_Y, dev_X, svm)
            errors[param.C, d].append(mse)
            cursor.update()
cursor.close()
best_C = None
best_mse = float("inf")
for d in range(1, D + 1):
    data = [errors[3**k, d] for k in range(-K, K + 1)]
    means = np.array([np.mean(d) for d in data])
    stds = np.array([np.std(d) for d in data])
    Cs = [3**k for k in range(-K, K + 1)]
    for C, mean in zip(Cs, means):
        if mean <= best_mse:
            best_C = C
            best_mse = mean
    plt.figure()
    plt.errorbar(Cs, means, yerr=stds, marker="o")
    plt.title(fR"Cross validation MSE by $C$ ($d={d}$)")
    plt.xscale("symlog")
    plt.xlabel(R"$C$")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()

# Part C.4
cursor = tqdm(total=D * n_splits)
val_errors = defaultdict(list)
test_errors = defaultdict(list)
n_vecs = defaultdict(list)
n_vecs_margin = defaultdict(list)
for d in range(1, D + 1):
    param = svm_parameter()
    param.C = best_C
    param.kernel_type = POLY
    param.degree = d
    for idx in range(n_splits):
        dev_X = split_X[idx]
        dev_Y = split_Y[idx]
        x = np.concatenate(split_X[:idx] + split_X[idx + 1:], axis=0)
        y = np.concatenate(split_Y[:idx] + split_Y[idx + 1:], axis=0)
        prob = svm_problem(y, x)
        svm = svm_train(prob, param)
        coeffs = svm.get_sv_coef()
        max_coeff = max(abs(t[0]) for t in coeffs)
        n_vecs[d].append(len(coeffs))
        n_vecs_margin[d].append(len([c for c in coeffs if abs(c[0]) == max_coeff]))
        p_label, (acc, mse, _), p_val = svm_predict(dev_Y, dev_X, svm)
        val_errors[d].append(mse)
        _, (_, mse, _), _ = svm_predict(test_Y, test_X, svm)
        test_errors[d].append(mse)
        cursor.update()
cursor.close()
ds = val_errors.keys()
val_means = [np.mean(data) for data in val_errors.values()]
val_stds = [np.std(data) for data in val_errors.values()]
test_means = [np.mean(data) for data in test_errors.values()]
test_stds = [np.std(data) for data in test_errors.values()]
_, best_d = min(zip(val_means, ds))
plt.figure()
plt.errorbar(ds, val_means, yerr=val_stds, label="crossval", marker="o")
plt.errorbar(ds, test_means, yerr=test_stds, label="test", marker="o")
plt.legend()
plt.title(Rf"Cross-validation and test MSE by $d$ ($C^* = {best_C}$)")
plt.xlabel(R"$d$")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

plt.figure()
n_vecs_mean = [np.mean(n_vecs[d]) for d in range(1, D + 1)]
n_vecs_margin_mean = [np.mean(n_vecs_margin[d]) for d in range(1, D + 1)]
n_vecs_std = [np.std(n_vecs[d]) for d in range(1, D + 1)]
n_vecs_margin_std = [np.std(n_vecs_margin[d]) for d in range(1, D + 1)]
plt.errorbar(ds, n_vecs_mean, yerr=n_vecs_std, label="total", marker="o")
plt.errorbar(ds, n_vecs_margin_mean, yerr=n_vecs_margin_std, label="marginal", marker="o")
plt.title("Number of support vectors and marginal support vectors")
plt.xlabel(R"$d$")
plt.ylabel("# support vectors")
plt.legend()
plt.tight_layout()
plt.show()

# Part C.5
param = svm_parameter()
param.C = best_C
param.kernel_type = POLY
param.degree = best_d
train_errors = []
test_errors = []
# Train data is already randomly shuffled, so we can just index it.
sample_sizes = np.linspace(10, n_train, 100)
for n_samples in tqdm(sample_sizes):
    n_samples = int(n_samples)
    x = train_X[:n_samples]
    y = train_Y[:n_samples]
    prob = svm_problem(y, x)
    svm = svm_train(prob, param)
    _, (_, train_mse, _), _ = svm_predict(y, x, svm)
    _, (_, test_mse, _), _ = svm_predict(test_Y, test_X, svm)
    train_errors.append(train_mse)
    test_errors.append(test_mse)
plt.figure()
plt.plot(sample_sizes, train_errors, label="train")
plt.plot(sample_sizes, test_errors, label="test")
plt.legend()
plt.xlabel("# samples")
plt.ylabel("MSE")
plt.title(Rf"MSE with optimal hyperparams ($C^* = {best_C}, d^* = {best_d}$)")
plt.tight_layout()
plt.show()
