import numpy as np

# 前向
x, y = 2.0, 3.0

a = x * y          # 6
b = x ** 2         # 4
c = a + b          # 10
f = c ** 2         # 100

# 反向（从 f 开始，梯度初始为 1）
grad_f = 1.0

grad_c = grad_f * 2 * c          # 20
grad_a = grad_c * 1.0            # 20
grad_b = grad_c * 1.0            # 20

grad_x_from_a = grad_a * y       # 60
grad_y        = grad_a * x       # 40
grad_x_from_b = grad_b * 2 * x   # 80

grad_x = grad_x_from_a + grad_x_from_b   # 140

print("f =", f)
print("df/dx =", grad_x)   # 140
print("df/dy =", grad_y)   # 40