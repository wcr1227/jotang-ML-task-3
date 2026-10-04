# PyTorch自动微分
import torch

# 创建需要梯度的张量
# requires_grad=True 告诉 PyTorch保存梯度
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向传播
a = x * y
b = x * x
c = a + b
f = c * c

# 反向传播
f.backward() # 自动反向传播

print(f"PyTorch 前向结果: f={f.item()}")
print(f"PyTorch 梯度: dx={x.grad.item()}, dy={y.grad.item()}")