def filters(df, filter_name):
    if filter_name == "小市值":
        df = df[df['总市值 （万元）'] < 300000]

    elif filter_name == "小市值_非财报月":
        df = df[df['总市值 （万元）'] < 300000]
        df['月份'] = df['交易日期'].dt.month  # 按照最后一个交易日在哪个月计算
        df = df[~df['月份'].isin([1, 4])]
        df.drop(columns='月份', inplace=True)  # 删除月份这一列

    else:
        raise Exception(f"过滤方法 '{filter_name}' 不存在。")

    return df
