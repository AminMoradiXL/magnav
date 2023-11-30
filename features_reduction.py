import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from matplotlib import pyplot as plt

flight_number = '1002' # 1002, 1003, 1004, 1005, 1006, 1007, 'all', 'all_except_1005'
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')

selected = ['mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
                     'diurnal',
                     'flux_b_x', 'flux_b_y', 'flux_c_y',
                     'ins_vw', 'ins_wander', 
                     'static_p','total_p',
                     'vol_srvo']


data_selected = data.loc[:, selected]

label = data.loc[:, ['utm_x', 'utm_y', 'utm_z']]

scaler = MinMaxScaler()
data_selected_n = pd.DataFrame(scaler.fit_transform(data_selected), columns=data_selected.columns)
label_n = pd.DataFrame(scaler.fit_transform(label), columns=label.columns)
label_n['dist'] = label_n['utm_x']**2 + label_n['utm_y']**2 + label_n['utm_z']**2

pca = PCA(n_components = 3)
components = pca.fit_transform(data_selected_n)

# kpca = KernelPCA(n_components=3, kernel= 'rbf')
# components = kpca.fit_transform(data_selected_n)

# iso = Isomap(n_jobs = -1)
# components = iso.fit_transform(data_selected_n)

# lle = LocallyLinearEmbedding(n_components= 3, n_jobs = -1)
# components = lle.fit_transform(data_selected_n)

# 3D

# fig = plt.figure(figsize = (15,15))
# ax = fig.add_subplot(projection = '3d')
# ax.grid()
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_axisbelow(True)
# p = ax.scatter(components[:,0], components[:,1], components[:,2], c = label_n['dist'], cmap = 'RdBu')
# ax.set_xlabel(r'$c_0$', fontsize = 32)
# ax.set_ylabel(r'$c_1$', fontsize = 32)
# ax.set_zlabel(r'$c_2$', fontsize = 32)
# ax.xaxis.labelpad = 20
# ax.yaxis.labelpad = 20
# ax.zaxis.labelpad = 20
# ax.set_aspect('equal')
# # ax.set_title('PCA components of selected features in 3d', fontsize = 36)
# ax.tick_params(labelsize = 20, pad = 10)
# cb = fig.colorbar(p, shrink = 0.6, location = 'left', pad=0)
# cb.set_label(label = 'Euclidean distance', fontsize = 32)
# cb.ax.tick_params(which = 'major', labelsize = 20)
# fig.savefig('Results/pca3d.png', dpi = 600)

# 2D

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot()
ax.grid()
ax.set_axisbelow(True)
p = ax.scatter(components[:,0], components[:,1], c = label_n['dist'], cmap = 'RdBu')
ax.set_xlabel(r'$c_0$', fontsize = 32)
ax.set_ylabel(r'$c_1$', fontsize = 32)
ax.xaxis.labelpad = 0
ax.yaxis.labelpad = 0
ax.tick_params(labelsize = 20, pad = 10)
ax.set_aspect('equal')
cb = fig.colorbar(p, shrink = 0.6, location = 'left', pad=0.15)
cb.set_label(label = 'Euclidean distance', fontsize = 32)
cb.ax.tick_params(which = 'major', labelsize = 20)
fig.savefig('Results/pca2d.png', dpi = 600)


