from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

min_depth = 10
max_depth = 30
depth_list = np.arange(min_depth, max_depth)
flight_number = '1002'
method_list = ['ins_aided','ins_tl_aided', 'ins_free']

log = pd.DataFrame()
test_count = 10

def drms(model, X_test, X_train, y_test, y_train):

    model.fit(X_train, y_train)     
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    error_test = y_pred_test - y_test
    MAE_test = np.mean(np.abs(error_test), axis=0)
    error_train = y_pred_train - y_train
    MAE_train = np.mean(np.abs(error_train), axis=0)
    DRMS_test = np.sqrt(0.5*(MAE_test.loc['utm_x']**2 + MAE_test.loc['utm_y']**2))
    DRMS_train = np.sqrt(0.5*(MAE_train.loc['utm_x']**2 + MAE_train.loc['utm_y']**2))
    
    return DRMS_test, DRMS_train

for i, depth in enumerate(depth_list):

    for j, method in enumerate(method_list):
        
        data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')
        
        scaler = MinMaxScaler()
        
        if method == 'ins_aided': 
            input_features = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
                                 'diurnal',
                                 'flux_b_x', 'flux_b_y', 'flux_c_y',
                                 'ins_vw', 'ins_wander', 
                                 'static_p','total_p',
                                 'vol_srvo']
        elif method == 'ins_free': 
            input_features = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
                                 'diurnal',
                                 'flux_b_x', 'flux_b_y', 'flux_c_y',
                                 # 'ins_vw', 'ins_wander', 
                                 'static_p','total_p',
                                 'vol_srvo']
        elif method == 'ins_tl_aided': 
            input_features = ['mag_3_c', 'mag_4_c', 'mag_5_c', 
                                 'diurnal',
                                 'flux_b_x', 'flux_b_y', 'flux_c_y',
                                 'ins_vw', 'ins_wander', 
                                 'static_p','total_p',
                                  'vol_srvo']
        
        data_in_df = data.loc[:, input_features]
        data_in_df = pd.DataFrame(scaler.fit_transform(data_in_df), columns=data.loc[:,input_features].columns)
        
        output_labels = ['utm_x','utm_y','utm_z']
       
        data_out_df = data.loc[:, output_labels]
        
        data_in_df = data.loc[:, input_features]
        
        X_train, X_test, y_train, y_test = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
        X_train, X_val, y_train, y_val = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
        
        forest_model = RandomForestRegressor(verbose=1, max_depth= depth, n_jobs = -1)
      
        DRMS_test, DRMS_train = drms(forest_model, X_test, X_train, y_test, y_train)
       
        log.loc[f'{method}, {depth}','depth'] = depth
        log.loc[f'{method}, {depth}','DRMS_test'] = DRMS_test
        log.loc[f'{method}, {depth}','DRMS_train'] = DRMS_train
    

log.to_pickle('Results/depth_comp_new.pkl')

# df = pd.read_pickle('Results/comparison_log.pkl')




