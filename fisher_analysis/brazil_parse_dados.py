import pandas as pd
import numpy  as np


f_name = 'fisher_analysis/dados_jun2018.csv'

brazil_df = pd.read_csv(f_name, index_col=['source', 'date'] , parse_dates=True, infer_datetime_format=True)

cantar = brazil_df.filter(like='sistemaCantareira', axis=0)
cantar = cantar.drop('daily_rainfall', axis=1)

cantar = cantar.reset_index(level=['source'])
# cantar1 = cantar.resample(rule='M').mean()
cantar2 = cantar.resample(rule='M').last()
# cantar = cantar.reset_index(level=['date'])
cantar2 = cantar2.reset_index(level=['date'])

cantar2.to_csv('fisher_analysis/cantar2.csv', index_label='index')
