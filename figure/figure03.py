import pickle
import numpy as np
import sacpy as scp
import pandas as pd
import xarray as xr
from scipy.stats import linregress
import re, sys, glob
sys.path.append('./library')
from cplot  import *
from cbasic import *

import cartopy.crs as ccrs
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['figure.dpi']  = 200
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

with open("./SST_bias_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
SST_bias_index = data['SST_bias_index']

with open("./scatter_LHF_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
LHF_index  = data['LHF_index']

with open("./scatter_Q925_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
Q925_index = data['Q925_index']

with open("./scatter_SLP_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
SLP_index  = data['SLP_index']

names  = ['CNLR', 'CNHR', 'ECLR', 'ECHR',
          'MPHR', 'MPXR', 'CMHR', 'CMVHR',
          'CELR', 'CEHR',
          'EWLR', 'EWMR', 'EWHR',
          'HGLL', 'HGMM', 'HGHM', 'HGHH']
labels = ['CNRM-CM6-1 (A250/O100)',       'CNRM-CM6-1-HR (A100/O25)',    'EC-Earth3P (A100/O100)',    'EC-Earth3P-HR (A50/O25)',
          'MPI-ESM1-2-HR (A100/O50)',     'MPI-ESM1-2-XR (A50/O50)',     'CMCC-CM2-HR4 (A100/O25)',   'CMCC-CM2-VHR4 (A25/O25)',
          'CESM1-CAM5-SE-LR (A100/O100)', 'CESM1-CAM5-SE-HR (A25/O10)',
          'ECMWF-IFS-LR (A50/O100)',      'ECMWF-IFS-MR (A50/O25)',      'ECMWF-IFS-HR (A25/O25)',
          'HadGEM3-GC31-LL (A250/O100)',  'HadGEM3-GC31-MM (A100/O25)',  'HadGEM3-GC31-HM (A50/O25)', 'HadGEM3-GC31-HH (A50/O10)']
marker_color_map = {
    'CNLR': ('o', 'blue'),    'CNHR':  ('s', 'blue'),    'ECLR': ('o', 'green'),   'ECHR':  ('s', 'green'),
    'MPHR': ('s', 'magenta'), 'MPXR':  ('d', 'magenta'), 'CMHR': ('s', 'purple'),  'CMVHR': ('d', 'purple'),
    'CELR': ('o', 'red'),     'CEHR':  ('p', 'red'),
    'EWLR': ('o', 'brown'),   'EWMR':  ('s', 'brown'),   'EWHR': ('d', 'brown'),
    'HGLL': ('o', 'orange'),  'HGMM':  ('s', 'orange'),  'HGHM': ('d', 'orange'),  'HGHH': ('p', 'orange')}

fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1,1], width_ratios=[1,1])

for i, (x_data, y_data, xlim, ylim, xticks, yticks) in enumerate([
    ([SST_bias_index[name] for name in names], [LHF_index[name].slope for name in names],               [-3.0,3.0], [0.0, 12.0], np.arange(-3.0, 3.1, 1.0),  np.arange(0.0,  12.1, 4.0)),
    ([SST_bias_index[name] for name in names], [Q925_index[name].slope*(86400/1004) for name in names], [-3.0,3.0], [-0.2, 0.4], np.arange(-3.0, 3.1, 1.0),  np.arange(-0.2, 0.41, 0.2)),
    ([SST_bias_index[name] for name in names], [SLP_index[name].slope for name in names],               [-3.0,3.0], [-1.6, 0.8], np.arange(-3.0, 3.1, 1.0),  np.arange(-1.6, 0.81, 0.8))]):
    ax = plt.subplot(gs[i])

    color_groups = {}
    for name in names:
        marker, color = marker_color_map[name]
        if color not in color_groups:
            color_groups[color] = {'x': [], 'y': []}
        color_groups[color]['x'].append(x_data[names.index(name)])
        color_groups[color]['y'].append(y_data[names.index(name)])

    for name, label in zip(names, labels):
        marker, color = marker_color_map[name]
        #ax.scatter(x_data[names.index(name)], y_data[names.index(name)], label=name+' '+label, marker=marker, color=color, edgecolor='k', s=133)
        ax.scatter(x_data[names.index(name)], y_data[names.index(name)], label=label, marker=marker, color=color, edgecolor='k', s=133)
    x = np.array(x_data); y = np.array(y_data)
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    ax.scatter(1000, 1000, color='None', label=" "); ax.scatter(1000, 1000, color='None', label=" ")

    p_value_sci = "{:.2e}".format(p_value); base, exponent = p_value_sci.split("e")
    textstr = f"R = {r_value:.2f}\nSlope = {slope:.2f}\nP = {float(base):.2f} × $10^{{{int(exponent)}}}$"
    
    if   i == 0: loc = 0.61; xlabel='KOE SST Bias (°C)'; ylabel='Reg. of LHF on KE index '+r'(W $\mathrm{m^{-2})}$'; xtick_minor = np.arange(-3.0, 3.1, 0.5); ytick_minor = np.arange(0.0, 12.1, 2.0);  ax.set_title('a', weight='bold', loc='left', fontsize=20)
    elif i == 1: loc = 0.61; xlabel='KOE SST Bias (°C)'; ylabel='Reg. of '+r'Q$_{925}$'+' on KE index '+r'(K $\mathrm{day^{-1})}$';     xtick_minor = np.arange(-3.0, 3.1, 0.5); ytick_minor = np.arange(-0.2, 0.41, 0.1); ax.set_title('b', weight='bold', loc='left', fontsize=20)
    elif i == 2: loc = 0.03; xlabel='KOE SST Bias (°C)'; ylabel='Reg. of SLP on KE index (hPa)';                    xtick_minor = np.arange(-3.0, 3.1, 0.5); ytick_minor = np.arange(-1.6, 0.61, 0.4); ax.set_title('c', weight='bold', loc='left', fontsize=20)
    ax.text(loc, 0.19, textstr, transform=ax.transAxes, fontsize=13, linespacing=1.3, verticalalignment='top')
    ax.plot(np.array(x_data), slope * np.array(x_data) + intercept, color='black')
    ax.set(xlim=xlim, xticks=xticks, ylim=ylim, yticks=yticks)
    ax.set_xlabel(xlabel, fontsize=15, labelpad=8); ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
    ax.tick_params(axis='x', which='major', length=5); ax.tick_params(axis='x', which='minor', length=3); ax.set_xticks(xtick_minor, minor=True)
    ax.tick_params(axis='y', which='major', length=5); ax.tick_params(axis='y', which='minor', length=3); ax.set_yticks(ytick_minor, minor=True)
    ax.axvline(x=np.array(x_data).mean(),c='grey',ls='--'); ax.axhline(y=np.array(y_data).mean(),c='grey',ls='--')

plt.tight_layout(w_pad=5, h_pad=3)
plt.legend(loc='lower right', bbox_to_anchor=(2.7, -0.02), ncols=2, fontsize=11, labelspacing=1.2, handletextpad=0, title="CMIP6 HighresMIP\n", title_fontsize=15, frameon=True)
plt.show()
