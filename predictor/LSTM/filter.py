
def filters(df, filter_name):

    if filter_name == "小市值":
        df = df[df['总市值 （万元）'] < 300000]

    else:
        raise Exception("尚无此过滤方法。")

    return df
