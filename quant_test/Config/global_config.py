# ====================全局配置====================

# ==========token配置==========
tushare_token = '30e6c0329269ab3e3ac6dfcc8737b274084e683ea121395597940bcc'
dingding_robot_id = '83ec1c5e05a0cfdbd76e7b0bf53feb98dec83633274b971c080ebf7e18bdd9b3'

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

# ==========黑名单配置==========
use_black_list = True
black_list = [
    "601628.SH",  # 中国人寿
]