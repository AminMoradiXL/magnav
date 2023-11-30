from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
import pickle
import pandas as pd

flight_number = '1002' # 1002, 1003, 1004, 1005, 1006, 1007, 'all', 'all_except_1005'
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')
data.dropna(inplace = True)
# data = data.dropna(axis = 0, subset = 'flux_a_t')

not_wanted_labels = ['line','mag_3_c','mag_4_c','mag_5_c','tt','N','slg','dt','ins_alt','ins_lat','ins_lon','lat','lon']
output_labels = ['utm_x','utm_y','utm_z']

input_features = list(data.columns)

for i in range(len(output_labels)): 
    input_features.remove(output_labels[i])

for i in range(len(not_wanted_labels)): 
    input_features.remove(not_wanted_labels[i])
 

data_out_df = data.loc[:, output_labels]

data_in_df = data.loc[:, input_features]

X_train, X_test, y_train, y_test = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
X_train, X_val, y_train, y_val = train_test_split(data_in_df, data_out_df, test_size=0.2, random_state=(0))
     

max_depth = 20

forest_model = ensemble.RandomForestRegressor(verbose = 1, max_depth = max_depth, n_jobs=-1)

sfs = SequentialFeatureSelector(forest_model, n_features_to_select = 10, n_jobs=-1)
sfs.fit(X_train, y_train)

filename = f'Model/sfs_{flight_number}.sav'
pickle.dump(sfs, open(filename, 'wb'))

# sfs = pickle.load(open(filename, 'rb'))

Selected = sfs.get_support()

sfs.transform(X_train).shape

sel_feat = []

for i in range(len(Selected)):
    if Selected[i]:
       sel_feat.append(X_test.columns[i]) 

    

