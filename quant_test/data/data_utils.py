import datetime


def get_current_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return current_date


def reorganize_industry_dct(industry_dct):
    industry_dct['申万三级行业'] = industry_dct['申万三级行业'][0] if len(industry_dct['申万三级行业']) > 0 else "无"
    industry_dct['申万二级行业'] = industry_dct['申万二级行业'][0] if len(industry_dct['申万二级行业']) > 0 else "无"
    industry_dct['申万一级行业'] = industry_dct['申万一级行业'][0] if len(industry_dct['申万一级行业']) > 0 else "无"
    return industry_dct