# 记录预测时间段与训练时间段
model_time_pair_dct = {
    ('2010-01-01', '2010-12-31'): ('2007-01-01', '2009-12-31'),
    ('2011-01-01', '2011-12-31'): ('2008-01-01', '2010-12-31'),
    ('2012-01-01', '2012-12-31'): ('2009-01-01', '2011-12-31'),
    ('2013-01-01', '2013-12-31'): ('2010-01-01', '2012-12-31'),
    ('2014-01-01', '2014-12-31'): ('2011-01-01', '2013-12-31'),
    ('2015-01-01', '2015-12-31'): ('2012-01-01', '2014-12-31'),
    ('2016-01-01', '2016-12-31'): ('2013-01-01', '2015-12-31'),
    ('2017-01-01', '2017-12-31'): ('2014-01-01', '2016-12-31'),
    ('2018-01-01', '2018-12-31'): ('2015-01-01', '2017-12-31'),
    ('2019-01-01', '2019-12-31'): ('2016-01-01', '2018-12-31'),
    ('2020-01-01', '2020-12-31'): ('2017-01-01', '2019-12-31'),
    ('2021-01-01', '2021-12-31'): ('2018-01-01', '2020-12-31'),
    ('2022-01-01', '2022-12-31'): ('2019-01-01', '2021-12-31'),
    ('2023-01-01', '2023-12-31'): ('2020-01-01', '2022-12-31'),
    # ('2024-01-01', '2024-12-31'): ('2021-01-01', '2023-12-31'),
    # ('2025-01-01', '2025-12-31'): ('2022-01-01', '2024-12-31'),
}

feature_li = []
period_type = 'W'
