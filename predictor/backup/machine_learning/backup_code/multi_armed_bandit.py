# https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA/
# 强化学习：多臂老虎机
# 将多个选股策略视为一个多臂老虎机，通过强化学习的方法确定哪个会带来更高的收益
# TODO：我认为可以直接基于基础策略的回测结果来执行强化学习策略，这样可以降低实现难度和资源消耗
import numpy as np
import matplotlib.pyplot as plt


# class BernoulliBandit:
#     """ 伯努利多臂老虎机,输入K表示拉杆个数 """
#     def __init__(self, K):
#         self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
#         self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
#         self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
#         self.K = K
#
#     def step(self, k):
#         # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
#         # 获奖）
#         if np.random.rand() < self.probs[k]:
#             return 1
#         else:
#             return 0
#
#
# class Solver:
#     """ 多臂老虎机算法基本框架 """
#     def __init__(self, bandit):
#         self.bandit = bandit
#         self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
#         self.regret = 0.  # 当前步的累积懊悔
#         self.actions = []  # 维护一个列表,记录每一步的动作
#         self.regrets = []  # 维护一个列表,记录每一步的累积懊悔
#
#     def update_regret(self, k):
#         # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
#         self.regret += self.bandit.best_prob - self.bandit.probs[k]
#         self.regrets.append(self.regret)
#
#     def run_one_step(self):
#         # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
#         raise NotImplementedError
#
#     def run(self, num_steps):
#         # 运行一定次数,num_steps为总运行次数
#         for _ in range(num_steps):
#             k = self.run_one_step()
#             self.counts[k] += 1
#             self.actions.append(k)
#             self.update_regret(k)
#
#
# class EpsilonGreedy(Solver):
#     """ epsilon贪婪算法,继承Solver类 """
#     def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
#         super(EpsilonGreedy, self).__init__(bandit)
#         self.epsilon = epsilon
#         #初始化拉动所有拉杆的期望奖励估值
#         self.estimates = np.array([init_prob] * self.bandit.K)
#
#     def run_one_step(self):
#         if np.random.random() < self.epsilon:
#             k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
#         else:
#             k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
#         r = self.bandit.step(k)  # 得到本次动作的奖励
#         self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
#         return k
#
#
# class UCB(Solver):
#     """ UCB算法,继承Solver类 """
#     def __init__(self, bandit, coef, init_prob=1.0):
#         super(UCB, self).__init__(bandit)
#         self.total_count = 0
#         self.estimates = np.array([init_prob] * self.bandit.K)
#         self.coef = coef
#
#     def run_one_step(self):
#         self.total_count += 1
#         ucb = self.estimates + self.coef * np.sqrt(
#             np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
#         k = np.argmax(ucb)  # 选出上置信界最大的拉杆
#         r = self.bandit.step(k)
#         self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
#         return k


class StrategyBandit:
    """
    策略老虎机
    """
    def __init__(self, candidate_strategy_tuple):
        self.candidate_strategy = candidate_strategy_tuple
        self.K = len(candidate_strategy_tuple)

    def step(self, k):
        print("选择的策略为：{}", self.candidate_strategy[k])
        # TODO：添加返回收益
        return


class StrategyChooser:
    """
    策略选择器
    """
    def __init__(self, strategy_bandit):
        self.bandit = strategy_bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_equity(self, k):
        # TODO:计算累计净值
        pass

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError


class EpsilonGreedy_chooser(StrategyChooser):
    """ epsilon贪婪算法,继承StrategyChooser类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy_chooser, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class UCB_chooser(StrategyChooser):
    """ UCB算法,继承StrategyChooser类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB_chooser, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
