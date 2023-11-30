from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import tensorflow as tf

fl_list = ['1002', '1003', '1004', '1005', '1006', '1007', 'all']
# fl_list = ['1002', '1003']

log = pd.DataFrame()

for i in enumerate(fl_list):

    flight_number = fl_list[i[0]] 
    
    data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')
    
    scaler = MinMaxScaler()
    
    input_features = ['mag_3_uc','mag_4_uc','mag_5_uc',
                      'diurnal','flux_b_x','flux_b_y','flux_b_z',
                      'ins_vw','ins_wander',
                      'static_p','total_p','vol_srvo'
                      ] # x,y,z are not input features
    data_in_df = data.loc[:, input_features]
    data_in_df = pd.DataFrame(scaler.fit_transform(data_in_df), columns=data.loc[:,input_features].columns)
    
    
    output_labels = ['slg']
    
    
    data_out_df = data.loc[:, output_labels]
    
    data_in_df = data.loc[:, input_features]
    
    X_train, X_test, y_train, y_test = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
    X_train, X_val, y_train, y_val = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
    
    def rmse(model, X_test, X_train, y_test, y_train, adddim = 0):
    
        model.fit(X_train, y_train)     
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        if adddim:
            y_pred_test = np.expand_dims(y_pred_test, axis = 1)
            y_pred_train = np.expand_dims(y_pred_train, axis = 1)
        error_test = y_pred_test - y_test
        RMSE_test = np.sqrt(np.mean(error_test**2))
        error_train = y_pred_train - y_train
        RMSE_train = np.sqrt(np.mean(error_train**2))
        RMSE_test = float(RMSE_test)
        RMSE_train = float(RMSE_train)
        return RMSE_test, RMSE_train
        
    max_depth = 30
    forest_model = RandomForestRegressor(verbose=1, max_depth= max_depth, n_jobs = -1)
    knn_model = KNeighborsRegressor(n_jobs = -1)
    tree_model = DecisionTreeRegressor(max_depth = 20)
    
    RMSE_test_f, RMSE_train_f = rmse(forest_model, X_test, X_train, y_test, y_train, adddim = 1)
    RMSE_test_k, RMSE_train_k = rmse(knn_model, X_test, X_train, y_test, y_train)
    RMSE_test_t, RMSE_train_t = rmse(tree_model, X_test, X_train, y_test, y_train, adddim = 1)
    
    
    log.loc[f'{flight_number}','RMSE_test_f'] = RMSE_test_f
    log.loc[f'{flight_number}','RMSE_train_f'] = RMSE_train_f
    log.loc[f'{flight_number}','RMSE_test_k'] = RMSE_test_k
    log.loc[f'{flight_number}','RMSE_train_k'] = RMSE_train_k
    log.loc[f'{flight_number}','RMSE_test_t'] = RMSE_test_t
    log.loc[f'{flight_number}','RMSE_train_t'] = RMSE_train_t


log.to_pickle('Results/comparison_slg_log.pkl')

df = pd.read_pickle('Results/comparison_slg_log.pkl')




