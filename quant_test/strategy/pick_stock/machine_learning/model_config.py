fcn_hidden_size = 64
fcn_hidden_layer = 1
out_performance = 0.05
fcn_patience = 50
total_epoch = 500
fcn_batch_size = 64
clf_lr = 0.0001
rgs_lr = 0.0001
l2_reg_coefficient = 0.0001
DROPOUT = 0.5
SHUFFLE = False
all_feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM', '股息率（%）', '股息率TTM（%）', '20日振幅', 'VWAP', '换手率（%）', '5日均线', '20日均线', 'bias_5', 'bias_20', '本周期涨跌幅']
feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比',
       '市盈率', '市净率', '换手率（%）', 'bias_5', 'bias_20', '本周期涨跌幅']
# feature_li = all_feature_li
model_time_pair_tpl = (('2020-01-01', '2020-04-30'),)

"""
回归模型最佳参数：
fcn_hidden_size = 64
fcn_hidden_layer = 1
out_performance = 0.05
fcn_patience = 50
total_epoch = 500
fcn_batch_size = 64
clf_lr = 0.0001
rgs_lr = 0.0001
l2_reg_coefficient = 0.0001
DROPOUT = 0.5
SHUFFLE = True
all_feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM', '股息率（%）', '股息率TTM（%）', '20日振幅', 'VWAP', '换手率（%）', '5日均线', '20日均线', 'bias_5', 'bias_20', '本周期涨跌幅']
feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比',
       '市盈率', '市净率', '换手率（%）', 'bias_5', 'bias_20', '本周期涨跌幅']
"""