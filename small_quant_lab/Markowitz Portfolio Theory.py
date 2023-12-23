# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 示例数据：三个资产的预期收益率和标准差
expected_returns = [0.12, 0.10, 0.07]  # 例如，12%, 10%, 7%
std_devs = [0.20, 0.15, 0.10]  # 例如，20%, 15%, 10%
num_assets = len(expected_returns)

# 生成一个简化的协方差矩阵（通常应基于实际数据生成）
cov_matrix = np.diag(std_devs) ** 2  # 为了简化，这里我们只使用对角矩阵

# 优化过程
def portfolio_variance(weights, cov_matrix):
    """ 计算组合方差 """
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, returns):
    """ 计算组合预期收益 """
    return np.sum(weights * returns)

# 确定有效边界
target_returns = np.linspace(min(expected_returns), max(expected_returns), 100)
target_risks = []

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)  # 权重之和为 1 的约束
bounds = tuple((0, 1) for _ in range(num_assets))  # 每个权重在 0 和 1 之间

for target_return in target_returns:
    # 为每个目标收益找到最小化风险的权重
    constraints_with_return = constraints + ({'type': 'eq', 'fun': lambda x: portfolio_return(x, expected_returns) - target_return},)
    result = minimize(portfolio_variance, num_assets * [1. / num_assets], args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints_with_return)
    if result.success:
        target_risks.append(np.sqrt(result.fun))

# 绘制有效边界
plt.figure(figsize=(10, 6))
plt.plot(target_risks, target_returns, 'b-', lw=2)
plt.title('Efficient Frontier of a Multi-Asset Portfolio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.grid(True)
plt.show()
