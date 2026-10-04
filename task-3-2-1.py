"""
实践2:两层神经网络的一次前向 + 反向传播（单样本）
结构:X → Linear → ReLU → Linear → Sigmoid → BCE Loss
对比:NumPy 手算梯度 vs PyTorch autograd
"""

import numpy as np
import torch
from sklearn.datasets import make_moons

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. 数据与参数
# ============================================================
# 单样本，2 维特征
X_all, y_all = make_moons(n_samples=100, noise=0.1, random_state=42)
X = X_all[0:1]   # 取1个样本
Y = y_all[0:1, None]

n_in, n_h, n_out = 2, 3, 1    # 隐藏层 3 个神经元

W1 = np.random.randn(n_h, n_in) * 0.5    # (3, 2)   3行（隐藏层神经元），2列（输入特征）
# W1 是输入层 → 隐藏层的权重矩阵，每一行，代表隐藏层里面一个神经元，和输入2个特征连接的权重
b1 = np.zeros((n_h,))                    # (3,)   隐藏层3个偏置
# 隐藏层每一个神经元，单独配一个偏置值
W2 = np.random.randn(n_out, n_h) * 0.5   # (1, 3)
# W2 是隐藏层 → 输出层的权重矩阵，这一行，代表输出层那1个神经元，和隐藏层3个神经元连接的权重
b2 = np.zeros((n_out,))                  # (1,)
# 输出层就1个神经元，所以只需要1个偏置

print("X :", X.shape, " Y :", Y.shape)
print("W1:", W1.shape, " b1:", b1.shape)
print("W2:", W2.shape, " b2:", b2.shape)

# ============================================================
# 2. 前向传播 (NumPy)
# ============================================================
# 激活函数sigmoid，把实数压缩到0～1，输出概率
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

EPS = 1e-12 # 防止 log(0)，避免出现负无穷

Z1   = X @ W1.T + b1          # (1,2)@(2,3)+(3,) = (1,3)
A1   = np.maximum(0, Z1)      # (1,3)  ReLU
Z2   = A1 @ W2.T + b2         # (1,3)@(3,1)+(1,) = (1,1)
Yhat = sigmoid(Z2)            # (1,1)

N = X.shape[0] 
L = -np.mean(Y * np.log(Yhat + EPS) + (1 - Y) * np.log(1 - Yhat + EPS)) # 二元交叉熵损失函数

print("\n[NumPy 前向]")
print("Z1  :", Z1.shape, Z1)
print("A1  :", A1.shape, A1)
print("Z2  :", Z2.shape, Z2)
print("Yhat:", Yhat.shape, Yhat)
print("Loss:", L)

# ============================================================
# 3. 反向传播 (NumPy, 链式法则从后向前)
# ============================================================
dZ2 = (Yhat - Y) / N        # (1,1)   BCE+Sigmoid 化简
dW2 = dZ2.T @ A1            # (1,1)@(1,3) = (1,3)
db2 = dZ2.sum(axis=0)       # (1,)
dA1 = dZ2 @ W2              # (1,1)@(1,3) = (1,3)
dZ1 = dA1 * (Z1 > 0)        # (1,3)  ReLU 门控
dW1 = dZ1.T @ X             # (3,1)@(1,2) = (3,2)
db1 = dZ1.sum(axis=0)       # (3,)
dX  = dZ1 @ W1              # (1,3)@(3,2) = (1,2)

print("\n[NumPy 反向]")
print("dZ2:", dZ2.shape, dZ2)
print("dW2:", dW2.shape, dW2)
print("db2:", db2.shape, db2)
print("dA1:", dA1.shape, dA1)
print("dZ1:", dZ1.shape, dZ1)
print("dW1:", dW1.shape, dW1)
print("db1:", db1.shape, db1)
print("dX :", dX.shape, dX)

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

print("\n[PyTorch 前向/反向]")
print("Loss:", loss_t.item())
print("dW2 :", W2_t.grad.shape, W2_t.grad)
print("db2 :", b2_t.grad.shape, b2_t.grad)
print("dW1 :", W1_t.grad.shape, W1_t.grad)
print("db1 :", b1_t.grad.shape, b1_t.grad)

# ============================================================
# 5. 逐项对比
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