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
from sklearn.decomposition import PCA 
from sklearn.manifold import LocallyLinearEmbedding

fl_list = ['1002', '1003', '1004', '1005', '1006', '1007', 'all', 'all_except_1005']
# fl_list = ['1002', '1003']

log = pd.DataFrame()

for i in enumerate(fl_list):

    flight_number = fl_list[i[0]] 
    
    data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')
    data.dropna(inplace = True)
    scaler = MinMaxScaler()
    
    # input_features = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
    #                      'diurnal',
    #                      'flux_b_x', 'flux_b_y', 'flux_c_y',
    #                      'ins_vw', 'ins_wander', 
    #                      'static_p','total_p',
    #                      'vol_srvo']
    # data_in_df = data.loc[:, input_features]

    output_labels = ['utm_x','utm_y','utm_z']
    not_wanted_labels = ['line','mag_3_c','mag_4_c','mag_5_c','tt','N','slg','dt','ins_alt','ins_lat','ins_lon','lat','lon']

    
    data_out_df = data.loc[:, output_labels]
    
    data_in_df = data.drop(columns = output_labels + not_wanted_labels)
    data_in_df = pd.DataFrame(scaler.fit_transform(data_in_df), columns=data_in_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
    X_train, X_val, y_train, y_val = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
    
    pca = PCA(n_components = 10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    
    # lle = LocallyLinearEmbedding(n_components = 5)
    # X_train = lle.fit_transform(X_train)
    # X_test = lle.fit_transform(X_test)
    
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
    
    max_depth = 30
    forest_model = RandomForestRegressor(verbose=1, max_depth= max_depth, n_jobs = -1)
    knn_model = KNeighborsRegressor(n_jobs = -1)
    tree_model = DecisionTreeRegressor(max_depth = 20)
    
    DRMS_test_f, DRMS_train_f = drms(forest_model, X_test, X_train, y_test, y_train)
    DRMS_test_k, DRMS_train_k = drms(knn_model, X_test, X_train, y_test, y_train)
    DRMS_test_t, DRMS_train_t = drms(tree_model, X_test, X_train, y_test, y_train)
    
    
    log.loc[f'{flight_number}','DRMS_test_f'] = DRMS_test_f
    log.loc[f'{flight_number}','DRMS_train_f'] = DRMS_train_f
    log.loc[f'{flight_number}','DRMS_test_k'] = DRMS_test_k
    log.loc[f'{flight_number}','DRMS_train_k'] = DRMS_train_k
    log.loc[f'{flight_number}','DRMS_test_t'] = DRMS_test_t
    log.loc[f'{flight_number}','DRMS_train_t'] = DRMS_train_t


log.to_pickle('Results/comparison_log_pca.pkl')





