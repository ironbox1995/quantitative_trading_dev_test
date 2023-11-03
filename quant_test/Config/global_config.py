# ====================全局配置====================

# ==========策略配置==========
Second_Board_available = False  # 创业板
STAR_Market_available = False  # 科创板
DAILY_PICK_TIME = False  # 按日线择时

# ==========路径配置==========
project_path = 'F:\quantitative_trading_dev_test\quant_test'

# ==========项目配置==========
dev_or_test = True
use_financial_data = False
force_run = True

# ==========买入配置==========
total_position = -1  # -1代表满仓

# ==========黑名单配置==========
use_black_list = True
black_list = [
    "002856.SZ",  # 美芝股份：此股涨少跌多，破坏高抛低吸
]