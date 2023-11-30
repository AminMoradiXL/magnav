import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import numpy as np

flight_number = '1002' # 1002, 1003, 1004, 1005, 1006, 1007, 'all', 'all_except_1005'
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')

scaler = MinMaxScaler()

input_features = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
                     'diurnal',
                     'flux_b_x', 'flux_b_y', 'flux_c_y',
                     # 'ins_vw', 'ins_wander', 
                     'static_p','total_p',
                     'vol_srvo']
data_in_df = data.loc[:, input_features]
data_in_df = pd.DataFrame(scaler.fit_transform(data_in_df), columns=data.loc[:,input_features].columns)

output_labels = ['utm_x','utm_y','utm_z']

data_out_df = data.loc[:, output_labels]

X_train, X_test, y_train, y_test = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
X_train, X_val, y_train, y_val = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))

    
data_perm = X_train.copy()
max_depth = 25
forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
forest_model.fit(data_perm, y_train) 
y_pred = forest_model.predict(X_test)
error = y_pred - y_test
MAE = np.mean(np.abs(error), axis=0)
baseline = np.sqrt(0.5*(MAE.loc['utm_x']**2 + MAE.loc['utm_y']**2))
print(f'baseline = {baseline}')

DRMS_record = []

for i in range(data_in_df.shape[1]):
    data_perm = X_train.copy()
    data_perm.iloc[:, i] = np.random.permutation(data_perm.iloc[:, i])
    max_depth = 25
    forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
    forest_model.fit(data_perm, y_train) 
    y_pred = forest_model.predict(X_test)
    error = y_pred - y_test
    MAE = np.mean(np.abs(error), axis=0)
    DRMS = np.sqrt(0.5*(MAE.loc['utm_x']**2 + MAE.loc['utm_y']**2))
    print(f'permuated feature = {i}, DRMS = {DRMS}')
    DRMS_record.append(DRMS) 

DRMS_dropped_record = []
for i in range(data_in_df.shape[1]):
    data_perm = X_train.copy()
    data_perm.drop(columns = data_perm.columns[i], inplace = True)
    max_depth = 25
    forest_model = RandomForestRegressor(max_depth= max_depth, n_jobs = -1)
    forest_model.fit(data_perm, y_train) 
    data_test_perm = X_test.copy()
    A = data_test_perm.drop(columns = data_test_perm.columns[i], inplace = False)
    
    y_pred = forest_model.predict(A)
    error = y_pred - y_test
    MAE = np.mean(np.abs(error), axis=0)
    DRMS = np.sqrt(0.5*(MAE.loc['utm_x']**2 + MAE.loc['utm_y']**2))
    print(f'dropped feature = {i}, DRMS = {DRMS}')
    DRMS_dropped_record.append(DRMS) 
    

   

fig, ax = plt.subplots(constrained_layout = True)
ax.bar(X_train.columns, DRMS_record, width=0.5, align='edge',label='Permuted', color = '#cb4b43')
ax.bar(X_train.columns, DRMS_dropped_record, width=0.5, align='center', label='Dropped', color = '#337eb8')
ax.set_xticklabels([it.split('(')[0].strip() for it in X_train.columns], rotation=75)
ax.hlines(baseline, 0, len(X_train.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Test DRMS (m)', fontsize = 'large')
ax.set_xlabel('Features', fontsize = 'large')
# ax.tick_params(labelsize = 15)
ax.grid()
ax.set_axisbelow(True)
ax.legend(fontsize = 'large')
fig.savefig('Results/feature_importance.png', dpi = 600)
