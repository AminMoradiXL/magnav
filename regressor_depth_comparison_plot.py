import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_pickle('Results/depth_comp.pkl')

ins_aided = data.iloc[0::3]
ins_tl_aided = data.iloc[1::3]
ins_free = data.iloc[2::3]

fig, ax = plt.subplots(constrained_layout = True)
ax.errorbar(ins_free['depth'], ins_free['DRMS_test'], marker='s', label= 'TL INS free (test)', color = '#cb4b43')
ax.errorbar(ins_aided['depth'], ins_aided['DRMS_test'], marker='s', label= 'INS aided (test)', color = '#337eb8')
# ax.errorbar(ins_free['depth'], ins_free['DRMS_train'], marker='s', label= 'TL INS free (train)', color = '#fac8af')
# ax.errorbar(ins_aided['depth'], ins_aided['DRMS_train'], marker='s', label= 'INS aided (train)', color = '#d2e6f0')

ax.set_ylabel('DRMS (m)', fontsize = 'large')
ax.set_xlabel('Random forest max depth', fontsize = 'large')
# ax.set_xticks(ind)
# ax.set_xticklabels(df.index)
ax.grid()
ax.set_axisbelow(True)
ax.legend(fontsize = 'large')

fig.tight_layout()
fig.savefig('Results/depth_comp.png', dpi = 600)
plt.show()


