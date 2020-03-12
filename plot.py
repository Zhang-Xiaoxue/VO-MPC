# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import time

import json

import iros_casadi_3 as res

# %%
sysParam = res.sysParam

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', \
    'black', 'steelblue', 'olive', 'pink', 'chocolate', 'orange', 'indigo']


# %%
if sysParam['isAvoid'] == False:
    with open('results_No_avoid_data.json', 'r') as fp:  
        results = json.loads(fp.read())
else:
    with open('case'+str(sysParam['case'])+'.json', 'r') as fp:  
        results = json.loads(fp.read())

# idx_host = 5
for idx_host in range(sysParam['num_robot']):
    locals()['x_result_'+str(idx_host)] = np.asarray(results['x_results_'+str(idx_host)]).T # (4, #)
    locals()['u_result_'+str(idx_host)] = np.asarray(results['u_results_'+str(idx_host)]).T # (2, #)
    locals()['xr_'+str(idx_host)] = np.asarray(results['x_reference_'+str(idx_host)])


# %%
""" 2D static figure """
import matplotlib
matplotlib.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
fig1 = plt.figure(1,figsize=(9,9))
ax1 = fig1.add_subplot(111)

for idx_host in range(sysParam['num_robot']):
    ax1.plot(locals()['x_result_'+str(idx_host)][0],locals()['x_result_'+str(idx_host)][1], color=colors[idx_host], linewidth=4)
    ax1.add_patch(patches.Circle( xy=(locals()['x_result_'+str(idx_host)][0][0],locals()['x_result_'+str(idx_host)][1][0]), color=colors[idx_host], radius=sysParam['radius']))
    ax1.plot(locals()['xr_'+str(idx_host)][0], locals()['xr_'+str(idx_host)][1], color=colors[idx_host], linestyle=':', linewidth=4)
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
# plt.legend(['Planned Traj.','Reference Traj.'],  loc='best')
plt.tight_layout()
# plt.show()
plt.savefig('result2D_case'+str(sysParam['case'])+'.pdf')


# %%
""" 3D figure """
import matplotlib.ticker as ticker
import matplotlib
import mpl_toolkits.mplot3d.art3d as art3d
matplotlib.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})

label_spacing = 10

# Setup the plot
if sysParam['num_robot'] == 6:
    fig3 = plt.figure(3, figsize=(9,9))
elif sysParam['num_robot'] == 12:
    fig3 = plt.figure(3, figsize=(9,9))
ax3 = fig3.add_subplot(111, projection='3d')    
ax3.view_init(elev=15, azim=65)
count = len(x_result_0[0])
times = np.linspace(0, sysParam['runtime'], count)
for k in range(len(times))[::2]:
    for idx_host in range(sysParam['num_robot']):
        p = patches.Circle( xy=(locals()['x_result_'+str(idx_host)][0,k],locals()['x_result_'+str(idx_host)][1,k]), \
            radius=sysParam['radius'], alpha=0.5, color=colors[idx_host])
        ax3.add_patch(p)
        art3d.pathpatch_2d_to_3d(p,z=times[k],zdir="z")
ax3 = plt.gca()
tick_spacing_1 = 2
tick_spacing_2 = 2
tick_spacing_3 = 2
ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_1))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_2))
ax3.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_3))
ax3.set_xlabel(r'$x$ [m]', labelpad=label_spacing)
ax3.set_ylabel(r'$y$ [m]', labelpad=label_spacing)
ax3.zaxis.set_rotate_label(False)
# ax3.zaxis.label.set_rotation(0)
ax3.set_zlabel(r'time [s]', labelpad=label_spacing, rotation=90)
if sysParam['num_robot'] == 6:
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-3 ,3)
    ax3.set_zlim(0, 10)
elif sysParam['num_robot'] == 12:
    ax3.set_xlim(-7, 7)
    ax3.set_ylim(-7 ,7)
    ax3.set_zlim(0, 10)
plt.tight_layout()
# plt.show()
plt.savefig('result3D_case'+str(sysParam['case'])+'.pdf')

#  %%
# """ 2D static wit radius figure """
# fig4 = plt.figure(4)
# ax4 = fig4.add_subplot(111)
# ax4.add_patch(patches.Rectangle( (-5,-3),10,6, linewidth=1, edgecolor='b', facecolor='none'))
# ax4.plot(locals()['xr_'+str(idx_host)][0], locals()['xr_'+str(idx_host)][1], color=colors[idx_host], linestyle=':')
# for k in range(len(x_result_0[0]))[::5]:
#     for idx_host in range(sysParam['num_robot']):
#         ax4.add_patch(patches.Circle( xy=(locals()['x_result_'+str(idx_host)][0,k],locals()['x_result_'+str(idx_host)][1,k]), \
#             radius=sysParam['radius'], alpha=0.5, color=colors[idx_host], edgecolor='none' ))
# plt.show()

# %%
