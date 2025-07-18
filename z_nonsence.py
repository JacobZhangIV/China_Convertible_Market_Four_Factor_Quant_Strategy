import rqdatac
import pandas as pd
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2020-12-31')
instrument_type = 'Convertible'
factor_name = '第二产业增加值占GDP比重(现价)'
remaining_time_to_mature ='短期 (1-3年)' #labels = ['短期 (1-3年)', '中期 (3-5年)', '长期 (5+年)']
rolling_window = 90
folder_name = f'econ_factor_{remaining_time_to_mature}'
profolio_name = ''

# 请确保您已经初始化了 rqdatac
rqdatac.init()

# 获取所有类型为'INDX'（指数）的合约
all_indices_df = rqdatac.all_instruments(type='INDX')

# 打印出DataFrame的前几行查看
print(all_indices_df.head())

df = rqdatac.get_price('000300.XSHG',start_date,end_date)
print(df)