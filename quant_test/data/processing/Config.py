"""
《邢不行-2021新版|Python股票量化投资课程》
author: 邢不行
微信: xbx9585

配置参数
"""
import os

# ===策略名
# strategy_name = '基础财务数据选股策略'

# ===复权配置
# fuquan_type = '后复权'

# ===选股参数设定
# period_type = 'M'  # W代表周，M代表月
# date_start = '2010-01-01'  # 需要从10年开始，因为使用到了ttm的同比差分，对比的是3年持续增长的数据
# date_end = '2022-01-10'
# select_stock_num = 3  # 选股数量
# c_rate = 1 / 10000  # 手续费
# t_rate = 1 / 1000  # 印花税

# # ===获取项目根目录
# _ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
# root_path = os.path.abspath(os.path.join(_, '../..'))  # 返回根目录文件夹
#
# # 导入财务数据路径
# finance_data_path = r'/historical/QMT_financial_data'

# 因为财务数据众多，将本策略中需要用到的财务数据字段罗列如下
# raw_fin_cols = [
#     # 短期借款 长期借款 应付债券 一年内到期的非流动负债
#     'B_st_borrow@xbx', 'B_lt_loan@xbx', 'B_bond_payable@xbx', 'B_noncurrent_liab_due_in1y@xbx',
#     # 营业总收入 负债应付利息 应付手续费及佣金
#     'R_operating_total_revenue@xbx', 'B_interest_payable@xbx', 'B_charge_and_commi_payable@xbx',
#     # 销售费用 管理费用 研发费用 资产减值损失
#     'R_sales_fee@xbx', 'R_manage_fee@xbx', 'R_rad_cost_sum@xbx', 'R_asset_impairment_loss@xbx',
#     # 固定资产折旧、油气资产折耗、生产性生物资产折旧 无形资产摊销 长期待摊费用摊销
#     'C_depreciation_etc@xbx', 'C_intangible_assets_amortized@xbx', 'C_lt_deferred_expenses_amrtzt@xbx',
#     # 其他综合利益 税金及附加 营业成本
#     'R_other_compre_income@xbx', 'R_operating_taxes_and_surcharge@xbx', 'R_operating_cost@xbx',
#     # 归母净利润 归母所有者权益合计 货币资金 流动负债合计
#     'R_np_atoopc@xbx', 'B_total_equity_atoopc@xbx', 'B_currency_fund@xbx', 'B_total_current_liab@xbx',
#     # 非流动负债合计 经营活动产生的现金流量净额
#     'B_total_noncurrent_liab@xbx', 'C_ncf_from_oa@xbx',
#     # 净利润  营业总成本
#     'R_np@xbx', 'R_operating_total_cost@xbx',
# ]

# 指定流量字段flow_fin_cols和截面字段cross_fin_cols。flow_fin_cols、cross_fin_cols必须是raw_fin_cols的子集
# flow_fin_cols = [
#     # 归母净利润 净利润 营业总收入 营业总成本
#     'R_np_atoopc@xbx', 'R_np@xbx', 'R_operating_total_revenue@xbx', 'R_operating_total_cost@xbx'
# ]

# raw_fin_cols财务数据中所需要计算截面数据的原生字段
cross_fin_cols = []

# 下面是处理财务数据之后需要的ttm，同比等一些字段
# derived_fin_cols = [
#     # 归母净利润_TTM  归母净利润_TTM同比  净利润_TTM  净利润_TTM同比
#     'R_np_atoopc@xbx_ttm', 'R_np_atoopc@xbx_ttm同比', 'R_np@xbx_ttm', 'R_np@xbx_ttm同比',
#     # 营业总收入_TTM  营业总成本_TTM
#     'R_operating_total_revenue@xbx_ttm', 'R_operating_total_cost@xbx_ttm'
# ]


raw_fin_cols = ['短期借款', '长期借款', '应付债券', '一年内到期的非流动负债', '营业总收入', '负债应付利息', '应付手续费及佣金', '销售费用', '管理费用', '研发费用', '资产减值损失', '固定资产折旧、油气资产折耗、生产性物资折旧', '无形资产摊销', '长期待摊费用摊销', '其他综合利益', '税金及附加', '营业成本', '归母净利润', '归母所有者权益合计', '货币资金', '流动负债合计', '非流动负债合计', '经营活动产生的现金流量净额', '净利润', '营业总成本']
flow_fin_cols = ['归母净利润', '净利润', '营业总收入', '营业总成本']
derived_fin_cols = ['归母净利润_ttm', '归母净利润_ttm同比', '净利润_ttm', '净利润_ttm同比', '营业总收入_ttm', '营业总成本_ttm']
