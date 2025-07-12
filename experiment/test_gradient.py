import torch
import matplotlib.pyplot as plt

# 定义要优化的函数 f(x) = (x - 3)²
def f(x):
    return (x - 3) ** 2

# 初始化变量（需要梯度追踪）
x = torch.tensor(0.0, requires_grad=True)  # 初始值设为0

# 设置学习率和迭代次数
learning_rate = 0.1
iterations = 20

# 存储优化过程中的x和f(x)值用于可视化
history_x = []
history_f = []

# 梯度下降优化过程
for i in range(iterations):
    # 计算函数值
    y = f(x)
    
    # 反向传播计算梯度
    y.backward()
    
    # 打印当前状态（使用.detach()避免梯度追踪）
    print(f"Iteration {i+1}: x = {x.item():.4f}, f(x) = {y.item():.4f}, gradient = {x.grad.item():.4f}")
    
    # 记录历史值
    history_x.append(x.detach().item())
    history_f.append(y.detach().item())
    
    # 更新参数（使用.data避免影响梯度计算）
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # 清零梯度（重要！）
    x.grad.zero_()

# 可视化优化过程
plt.figure(figsize=(10, 5))
plt.plot(history_x, label='x value')
plt.plot(history_f, label='f(x) value')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()