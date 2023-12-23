# coding=utf-8
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设定投资标的 A 和 B 的参数
# R_A = 0.08  # A 的年预期收益率
# sigma_A = 0.15  # A 的年标准差
# R_B = 0.12  # B 的年预期收益率
# sigma_B = 0.20  # B 的年标准差
# rho_AB = 0.5  # A 和 B 之间的相关系数
#
# # 计算不同权重组合的预期收益率和风险
# weights = np.linspace(0, 1, 100)  # A 的权重从 0 到 1
# portfolio_returns = []
# portfolio_risks = []
#
# for w in weights:
#     # 计算组合预期收益率
#     R_portfolio = w * R_A + (1 - w) * R_B
#     portfolio_returns.append(R_portfolio)
#
#     # 计算组合风险
#     sigma_portfolio = np.sqrt((w * sigma_A)**2 + ((1 - w) * sigma_B)**2 +
#                               2 * w * (1 - w) * sigma_A * sigma_B * rho_AB)
#     portfolio_risks.append(sigma_portfolio)
#
# # 绘制有效边界
# plt.figure(figsize=(10, 6))
# plt.plot(portfolio_risks, portfolio_returns, 'b-', lw=2)
# plt.title('Efficient Frontier of Two Asset Portfolio')
# plt.xlabel('Portfolio Risk (Standard Deviation)')
# plt.ylabel('Portfolio Expected Return')
# plt.grid(True)
# plt.show()

# =========================多资产==============================

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，以便结果可复现
np.random.seed(42)

# 假设有三个投资标的
num_assets = 3

# 假设预期收益率和标准差
returns = np.random.normal(loc=0.1, scale=0.15, size=num_assets)
std_devs = np.random.normal(loc=0.2, scale=0.05, size=num_assets)

# 生成随机协方差矩阵
cov_matrix = np.random.rand(num_assets, num_assets)
cov_matrix = cov_matrix @ cov_matrix.T  # 使其为对称正定矩阵

# 蒙特卡洛模拟
num_portfolios = 10000
portfolio_returns = []
portfolio_risks = []

for _ in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # 确保权重之和为1

    # 计算组合预期收益率和风险
    R_portfolio = np.sum(weights * returns)
    sigma_portfolio = np.sqrt(weights.T @ cov_matrix @ weights)

    portfolio_returns.append(R_portfolio)
    portfolio_risks.append(sigma_portfolio)

# 绘制有效边界
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_risks, portfolio_returns, c='blue', marker='o')
plt.title('Efficient Frontier of a Multi-Asset Portfolio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.grid(True)
plt.show()
