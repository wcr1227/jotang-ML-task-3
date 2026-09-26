"""
实践2 · 作业点4:小 batch (N=4) 的前向 + 反向
结构:X → Linear → ReLU → Linear → Sigmoid → BCE
对比:NumPy 手算梯度 vs PyTorch autograd
重点：观察单样本 → batch 时，哪些公式和形状变了
"""

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. 数据与参数
# ============================================================
X = np.random.randn(4, 2)                 # (4, 2)  4个样本，2维
Y = np.array([[1.], [0.], [1.], [0.]])    # (4, 1)
N = X.shape[0]

n_in, n_h, n_out = 2, 3, 1

W1 = np.random.randn(n_h, n_in) * 0.5     # (3, 2)
b1 = np.zeros(n_h)                        # (3,)
W2 = np.random.randn(n_out, n_h) * 0.5    # (1, 3)
b2 = np.zeros(n_out)                      # (1,)

print("X :", X.shape, " Y :", Y.shape, " N =", N)
print("W1:", W1.shape, " b1:", b1.shape)
print("W2:", W2.shape, " b2:", b2.shape)

# ============================================================
# 2. 前向传播
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

EPS = 1e-12

Z1   = X @ W1.T + b1          # (4,2)@(2,3)+(3,) = (4,3)
A1   = np.maximum(0, Z1)      # (4,3)
Z2   = A1 @ W2.T + b2         # (4,3)@(3,1)+(1,) = (4,1)
Yhat = sigmoid(Z2)            # (4,1)
L    = -np.mean(Y * np.log(Yhat + EPS) + (1 - Y) * np.log(1 - Yhat + EPS))

print("\n[NumPy 前向]")
print("Z1  :", Z1.shape)
print("A1  :", A1.shape)
print("Z2  :", Z2.shape)
print("Yhat:", Yhat.shape)
print("Loss:", L)

# ============================================================
# 3. 反向传播
# ============================================================
dZ2 = (Yhat - Y) / N          # (4,1)  ← 注意 /N（单样本时 N=1 看不出）
dW2 = dZ2.T @ A1              # (1,4)@(4,3) = (1,3)  形状不变
db2 = dZ2.sum(axis=0)         # (1,)    ← 在 N 维上求和
dA1 = dZ2 @ W2                # (4,1)@(1,3) = (4,3)
dZ1 = dA1 * (Z1 > 0)          # (4,3)
dW1 = dZ1.T @ X               # (3,4)@(4,2) = (3,2)  形状不变
db1 = dZ1.sum(axis=0)         # (3,)    ← 在 N 维上求和
dX  = dZ1 @ W1                # (4,3)@(3,2) = (4,2)

print("\n[NumPy 反向]")
print("dZ2:", dZ2.shape, " dW2:", dW2.shape, " db2:", db2.shape)
print("dA1:", dA1.shape, " dZ1:", dZ1.shape)
print("dW1:", dW1.shape, " db1:", db1.shape, " dX :", dX.shape)

# ============================================================
# 4. PyTorch 复现
# ============================================================
X_t  = torch.tensor(X, dtype=torch.float32)
Y_t  = torch.tensor(Y, dtype=torch.float32)
W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

Z1_t   = X_t @ W1_t.T + b1_t
A1_t   = torch.relu(Z1_t)
Z2_t   = A1_t @ W2_t.T + b2_t
Yhat_t = torch.sigmoid(Z2_t)
loss_t = torch.nn.functional.binary_cross_entropy(Yhat_t, Y_t)
loss_t.backward()

print("\n[PyTorch]")
print("Loss:", loss_t.item())
print("dW1:", W1_t.grad.shape, " db1:", b1_t.grad.shape)
print("dW2:", W2_t.grad.shape, " db2:", b2_t.grad.shape)

# ============================================================
# 5. 梯度对比
# ============================================================
print("\n[梯度对比]")
def compare(name, np_grad, torch_grad):
    tg = torch_grad.detach().numpy()
    diff = np.abs(np_grad - tg)
    ok = np.allclose(np_grad, tg, atol=1e-6)
    print(f"{name:5s} | np{np_grad.shape} torch{tg.shape} "
          f"| max|Δ|={diff.max():.3e} | 一致={ok}")

compare("dW1", dW1, W1_t.grad)
compare("db1", db1, b1_t.grad)
compare("dW2", dW2, W2_t.grad)
compare("db2", db2, b2_t.grad)

print(f"\nLoss 对比: NumPy={L:.8f}  PyTorch={loss_t.item():.8f} "
      f"| Δ={abs(L - loss_t.item()):.3e}")