import matplotlib.pyplot as plt
import numpy as np 
import h5py
import pandas as pd
import seaborn as sb


flight_number = '1002' # 1002, 1003, 1004, 1005, 1006, 1007, 'all', 'all_except_1005'
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data.dropna(inplace = True)
data_n = scalar.fit_transform(data)
mean = data_n.mean(axis = 0)
std = data_n.std(axis = 0)


# input_features = ['flux_c_t','flux_c_z','cur_ac_lo','ins_alt'
#                   ,'vol_back_n','vol_back_p','vol_acc_n','vol_acc_p'
#                   ,'ins_lat', 'ins_roll'
#                   ,'mag_3_c', 'mag_4_c', 'mag_5_c'
#                   # ,'utm_x','utm_y','utm_z'
#                   ] # x,y,z are not input features

# output_labels = ['utm_x','utm_y','utm_z']

# data = data.drop(['N','dt','flight'], axis = 1)
# cor_matrix_x = data.corr()['utm_x'].abs().sort_values(ascending=False)
# cor_matrix_y = data.corr()['utm_y'].abs().sort_values(ascending=False)
# cor_matrix_z = data.corr()['utm_z'].abs().sort_values(ascending=False)

# upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))

# df1 = data.drop(data.columns[to_drop], axis=1)


# sb.heatmap(correlation, cmap="Blues", annot=True)

# correlation = correlation.dropna(axis = 1)
# abscorr = correlation.abs()
# a = abscorr.ge(0.5)

xyz = ['utm_x','utm_y','utm_z']
mag = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc']
current = ['cur_ac_hi', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'cur_strb',
 'cur_srvo_o', 'cur_srvo_m', 'cur_srvo_i', 'cur_acpwr', 'cur_outpwr', 'cur_bat_1', 'cur_bat_2']
voltage = ['vol_acpwr', 'vol_outpwr', 'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n',
            'vol_back_p', 'vol_back_n', 'vol_gyro_1', 'vol_gyro_2', 'vol_acc_p', 'vol_acc_n',
            'vol_block', 'vol_back', 'vol_servo', 'vol_cabt', 'vol_fan'] 
flux = ['flux_a_x', 'flux_a_y', 'flux_a_z', 'flux_a_t', 
        'flux_b_x', 'flux_b_y', 'flux_b_z', 'flux_b_t', 
        'flux_c_x', 'flux_c_y', 'flux_c_z', 'flux_c_t', 
        'flux_d_x', 'flux_d_y', 'flux_d_z', 'flux_d_t']

# selected = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
#             'cur_tank', 
#             'ins_wander', 'ins_lat', 'ins_lon','ins_alt',
#             'flux_a_t', 'flux_b_t', 'flux_d_t',
#             'vol_back', 'vol_res_n', 'vol_acc_p', 'vol_acc_n']

selected = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
                     'diurnal',
                     'flux_b_x', 'flux_b_y', 'flux_c_y',
                     'ins_vw', 'ins_wander', 
                     'ins_lon', 'ins_lat', 'ins_alt',
                     'static_p','total_p',
                     'vol_srvo']

ins = ['ins_acc_x', 'ins_acc_y', 'ins_acc_z', 'ins_wander',
       'ins_lat', 'ins_lon', 'ins_alt', 'ins_vn', 'ins_vw', 'ins_vu']

new_flight_selected = ['UNCOMPMAG3', 'UNCOMPMAG4', 'UNCOMPMAG5', 
            'CUR_TANK', 
            'FLUXA_TOT', 'FLUXB_TOT', 'FLUXD_TOT',
            'V_BACK', 'V_RESn', 'V_RESp', 'V_ACCn']

df_current = data.filter(xyz + current, axis = 1)
df_voltage = data.filter(xyz + voltage, axis = 1)
df_mag = data.filter(xyz + mag, axis = 1)
df_flux = data.filter(xyz + flux, axis = 1)
df_ins = data.filter(xyz + ins, axis = 1)
df_selected = data.filter(xyz + selected, axis = 1)

# correlation_mag = df_mag.corr() 
# fig, ax = plt.subplots(figsize=(12,12)) 
# sb.heatmap(correlation_mag, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('correlation of magnetic sensors with WGS xyz coordinates', fontsize = 14)
# fig.savefig('Results/correlation_mag.png', dpi = 600)

# correlation_current = df_current.corr() 
# fig, ax = plt.subplots(figsize=(12,12)) 
# sb.heatmap(correlation_current, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('correlation of current sensors with WGS xyz coordinates', fontsize = 14)
# fig.savefig('Results/correlation_cur.png', dpi = 600)

# correlation_voltage = df_voltage.corr() 
# fig, ax = plt.subplots(figsize=(12,12)) 
# sb.heatmap(correlation_voltage, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('correlation of voltage sensors with WGS xyz coordinates', fontsize = 14)
# fig.savefig('Results/correlation_vol.png', dpi = 600)

# correlation_flux = df_flux.corr() 
# fig, ax = plt.subplots(figsize=(12,12)) 
# sb.heatmap(correlation_flux, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('correlation of fluxgate magneto meters with WGS xyz coordinates', fontsize = 14)
# fig.savefig('Results/correlation_flx.png', dpi = 600)

# correlation_ins = df_ins.corr() 
# fig, ax = plt.subplots(figsize=(12,12)) 
# sb.heatmap(correlation_ins, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
# ax.set_title('correlation of INS sensors with WGS xyz coordinates', fontsize = 14)
# fig.savefig('Results/correlation_ins.png', dpi = 600)

correlation_selected = df_selected.corr() 
fig, ax = plt.subplots(figsize=(18,13)) 
sb.heatmap(correlation_selected, cmap='RdBu', annot=True, vmin = -1, vmax = 1, ax = ax)
ax.set_title('Correlation of selected sensors with WGS xyz coordinates\n', fontsize = 24)
ax.tick_params(labelsize = 18)
fig.savefig('Results/correlation_sel.png', dpi = 600)
