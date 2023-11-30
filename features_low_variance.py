import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

flight_number = '1002' # 1002, 1003, 1004, 1005, 1006, 1007, 'all', 'all_except_1005'
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

orig_std = data.std()
# sckd_mean = scaled_data.mean()
scld_std = scaled_data.std()

