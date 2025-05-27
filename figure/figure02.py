import pickle
import numpy as np
import sacpy as scp
import pandas as pd
import xarray as xr
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

def data_read(type, name, var, varname, region=[-90, 90, 0, 360]):
    file_pattern = f"../../data/{type}/{name}/{name}.{var}*"
    file = glob.glob(file_pattern)[0]
    
    ds   = xr.open_dataset(file)
    rename_dict = {}
    if 'latitude' in ds.dims:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.dims:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        ds = ds.rename(rename_dict)
    match = re.search(r'(\d{4})to(\d{4})', file)
    if match:
        start_year, end_year = match.groups()
        time_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='M')
        ds = ds.assign_coords(time=time_range)
    
    latS, latN, lonW, lonE = region
    data = ds[f"{varname}"].sortby('lat').sel(lat=slice(latS, latN), lon=slice(lonW, lonE))
    return data

syear = '1993'; eyear = '2022'; region1 = [25, 45, 110, 190]; region2 = [15, 65, 100, 260]
SSH = data_read('OBS',       'AVISO', 'SSH', 'adt', region1).sel(time=slice(syear,eyear))
LHF = data_read('OBS',    'J-OFURO3', 'LHF', 'LHF', region2).sel(time=slice(syear,eyear))
SLP = data_read('REANAL',     'ERA5', 'SLP', 'msl', region2).sel(time=slice(syear,eyear))/100.

_, _, dtr_anom_SSH = calc_anomaly(SSH)
_, _, dtr_anom_LHF = calc_anomaly(LHF)
_, _, dtr_anom_SLP = calc_anomaly(SLP)

index   = extract_index(dtr_anom_SSH, 31, 36, 140, 165)
KOE_LHF = extract_index(dtr_anom_LHF, 31, 45, 140, 175)
NP_SLP  = extract_index(dtr_anom_SLP, 35, 60, 140, 235)

syear = '1993'; eyear = '2022'
index1 = norm(season(extract_index(dtr_anom_SSH, 31, 36, 140, 165), [9,10,11]))[:-1]
index2 = season(KOE_LHF.sel(time=slice(syear,eyear)), [12,1])
index3 = season(NP_SLP.sel(time=slice(syear,eyear)), [12,1])

REG_LHF = scp.LinReg(index1, index2)
REG_SLP = scp.LinReg(index1, index3)

with open("./scatter_LHF_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
LHF_index  = data['LHF_index']

def data_read(group, name, var, varname, region=[-90, 90, 0, 360]):
    file_pattern = f"../../data/MODEL/HighresMIP/{group}/{name}.{var}*"
    file = glob.glob(file_pattern)[0]

    ds   = xr.open_dataset(file)
    match = re.search(r'(\d{4})to(\d{4})', file)
    if match:
        start_year, end_year = match.groups()
        time_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='M')
        ds = ds.assign_coords(time=time_range)

    latS, latN, lonW, lonE = region
    data = ds[f"{varname}"].sortby('lat').sel(lat=slice(latS, latN), lon=slice(lonW, lonE))
    return data

def data_read2(group, name, var, varname):
    file_pattern = f"../../data/MODEL/HighresMIP/{group}/{name}.{var}*"
    file = glob.glob(file_pattern)[0]

    ds   = xr.open_dataset(file)
    match = re.search(r'(\d{4})to(\d{4})', file)
    if match:
        start_year, end_year = match.groups()
        time_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='M')
        ds = ds.assign_coords(time=time_range)

    data = ds[f"{varname}"].sortby('lat')
    return data

with open("./KE_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
KE_axis = data['KE_axis']
KE_index = data['KE_index']

syear = '1950'; eyear = '2014'; region = [-20, 80, 100, 360]
CEHR_SST  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-HR', 'SST',  'TEMP', region).sel(time=slice(syear, eyear))
CEHR_SLP  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-HR', 'PSL',  'PSL',  region).sel(time=slice(syear,eyear))/100.
CEHR_Q925 = data_read2('CESM1-CAM5-SE', 'CESM1-CAM5-SE-HR', 'Q925', 'Q1').sel(time=slice(syear,eyear)).squeeze()
CEHR_Q925 = np2xr(CEHR_SLP, CEHR_Q925)
_, _, dtr_anom_CEHR_SLP = calc_anomaly(CEHR_SLP.sel(lon=slice(100,260),lat=slice(10,70)))
CEHR_Q925['lat'] = CEHR_SST['lat']; CEHR_Q925 = CEHR_Q925.where(~np.isnan(CEHR_SST))*((86400/1004))
_, _, dtr_anom_CEHR_Q925 = calc_anomaly(CEHR_Q925.sel(lon=slice(100,260),lat=slice(10,70)))

CMVHR_SST = data_read('CMCC-CM2',   'CMCC-CM2-VHR4', 'SST',  'tos', region).sel(time=slice(syear, eyear))
CMVHR_SLP = data_read('CMCC-CM2',   'CMCC-CM2-VHR4', 'PSL',  'psl', region).sel(time=slice(syear,eyear))/100.
CMVHR_Q925= data_read2('CMCC-CM2',  'CMCC-CM2-VHR4', 'Q925', 'Q1').sel(time=slice(syear,eyear)).squeeze()
CMVHR_Q925= np2xr(CMVHR_SLP, CMVHR_Q925)
_, _, dtr_anom_CMVHR_SLP = calc_anomaly(CMVHR_SLP.sel(lon=slice(100,260),lat=slice(10,70)))
CMVHR_Q925['lat'] = CMVHR_SST['lat']; CMVHR_Q925 = CMVHR_Q925.where(~np.isnan(CMVHR_SST))*((86400/1004))
_, _, dtr_anom_CMVHR_Q925 = calc_anomaly(CMVHR_Q925.sel(lon=slice(100,260),lat=slice(10,70)))

fnames = ['OBS', 'HighresMIP']
mnames = [
    'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth3P', 'EC-Earth3P-HR',
    'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'CMCC-CM2-HR4', 'CMCC-CM2-VHR4',
    'CESM1-CAM5-SE-LR', 'CESM1-CAM5-SE-HR',
    'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-HH',
    'ECMWF-IFS-LR', 'ECMWF-IFS-MR', 'ECMWF-IFS-HR']
categories = fnames + mnames

models = list(LHF_index.keys())
model_slopes = [LHF_index[model].slope for model in models]
model_p_values = [LHF_index[model].p_value for model in models]

slopes_all = [REG_LHF.slope, np.mean(model_slopes)] + model_slopes
p_all = [REG_LHF.p_value, np.mean(model_p_values) + 1] + model_p_values
models_std = np.std(model_slopes)
std_all = [0, models_std] + [0] * len(model_slopes)

categories = ['OBS', 'HighresMIP'] + models
colors_all = ['grey', 'tab:orange'] + ['tab:blue'] * len(models)
model_part = list(zip(models, model_slopes, model_p_values, [0] * len(models), colors_all[2:]))
model_part_sorted = sorted(model_part, key=lambda x: x[1], reverse=True)

categories_sorted = ['OBS', 'HighresMIP'] + [m[0] for m in model_part_sorted]
slopes_sorted = [REG_LHF.slope, np.mean(model_slopes)] + [m[1] for m in model_part_sorted]
p_sorted = [REG_LHF.p_value, np.mean(model_p_values) + 1] + [m[2] for m in model_part_sorted]
std_sorted = [0, models_std] + [m[3] for m in model_part_sorted]
colors_sorted = ['grey', 'tab:orange'] + [m[4] for m in model_part_sorted]

shortnames = [
    'CNLR', 'CNHR', 'ECLR', 'ECHR', 'MPHR', 'MPXR',
    'CMHR', 'CMVHR', 'CELR', 'CEHR', 'HGLL', 'HGMM',
    'HGHM', 'HGHH', 'EWLR', 'EWMR', 'EWHR']
name_map = dict(zip(shortnames, mnames))
name_map.update({'OBS': 'OBS', 'HighresMIP': 'HighresMIP'})
xtick_labels = [name_map.get(name, name) for name in categories_sorted]

axis   = KE_axis['CEHR'];  x_KOE,y_KOE = KOE_domain(axis)
index  = norm(season(KE_index['CEHR'].sel(time=slice(syear,eyear)),  [9,10,11]))[:-1]
target = season(dtr_anom_CEHR_Q925.sel(time=slice(syear,eyear)), [12,1]); CEHR_DBH_reg = scp.LinReg(index, target)
target = season(dtr_anom_CEHR_SLP.sel(time=slice(syear,eyear)), [12,1]);  CEHR_SLP_reg = scp.LinReg(index, target)

axis   = KE_axis['CMVHR']; x_KOE,y_KOE = KOE_domain(axis)
index  = norm(season(KE_index['CMVHR'].sel(time=slice(syear,eyear)), [9,10,11]))[:-1]
target = season(dtr_anom_CMVHR_Q925.sel(time=slice(syear,eyear)), [12,1]); CMVHR_DBH_reg = scp.LinReg(index, target)
target = season(dtr_anom_CMVHR_SLP.sel(time=slice(syear,eyear)), [12,1]);  CMVHR_SLP_reg = scp.LinReg(index, target)

fig = plt.figure(figsize=(9,10))
gs  = gridspec.GridSpec(nrows=5, ncols=2, height_ratios=[1,0.03,1,1,0.01], width_ratios=[1,1])

ax  = plt.subplot(gs[0,:]); ax.set_title('a', weight='bold', loc='left', fontsize=17)
siglev = 0.1; bars = []; colors = ['grey'] * 1 + ['tab:orange'] * 1 + ['tab:blue'] * 1 + ['tab:blue'] * 17
for i, (cat, slope, p, std) in enumerate(zip(categories_sorted, slopes_sorted, p_sorted, std_sorted)):
    hatch = '///' if p < siglev else None
    bar = ax.bar(cat, slope, color=colors_sorted[i], width=0.5, hatch=hatch, edgecolor='black', linewidth=0.5)
    if cat in ['OBS', 'HighresMIP']:
        ax.errorbar(cat, slope, yerr=std, fmt='none', ecolor='k', capsize=5, elinewidth=1)

xtick_labels = [name_map.get(name, name) for name in categories_sorted]
ax.set_xlim([-1,19])
ax.set_xticks(np.arange(len(categories_sorted)))
ax.set_xticklabels(xtick_labels, fontsize=11, rotation=40, ha='right')
ax.set_ylim([-1.5,12])
ax.set_yticks(np.arange(0,12.1,3))
ax.set_yticks(np.arange(-1.5,12.1,1.5), minor=True)
ax.set_ylabel(r'(W m$^{-2}$)', fontsize=13, labelpad=10)
ax.axhline(y=0, linestyle='--', c='grey', alpha=0.5)
ax.axvline(x=1.5, c='r', ls='--')

projection_map = ccrs.PlateCarree(central_longitude=180); transform_map = ccrs.PlateCarree(); domain = [18, 62, 110, 250]

cmap = cmap_white_center(plt.cm.PiYG_r);  levs = np.arange(-0.6,0.61,0.06)
ax = plt.subplot(gs[2,0], projection=projection_map); ax.set_title('b', weight='bold', loc='left', fontsize=17)
cf1= reg_map2(ax, CEHR_DBH_reg, domain, cmap, levs, transform_map)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 10)
ccrs_plot(ax, x_KOE.min(), x_KOE.max(), y_KOE.min(), y_KOE.max())

cmap = cmap_white_center(plt.cm.coolwarm); levs = np.arange(-1.2,1.21,0.12)
ax = plt.subplot(gs[2,1], projection=projection_map); ax.set_title('c', weight='bold', loc='left', fontsize=17)
cf2= reg_map2(ax, CEHR_SLP_reg, domain, cmap, levs, transform_map)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 10)
ccrs_plot(ax, 140, 235, 35, 60)

cmap = cmap_white_center(plt.cm.PiYG_r);  levs = np.arange(-0.6,0.61,0.06)
ax = plt.subplot(gs[3,0], projection=projection_map); ax.set_title('d', weight='bold', loc='left', fontsize=17)
cf1= reg_map2(ax, CMVHR_DBH_reg, domain, cmap, levs, transform_map)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 10)
ccrs_plot(ax, x_KOE.min(), x_KOE.max(), y_KOE.min(), y_KOE.max())

cmap = cmap_white_center(plt.cm.coolwarm); levs = np.arange(-1.2,1.21,0.12)
ax = plt.subplot(gs[3,1], projection=projection_map); ax.set_title('e', weight='bold', loc='left', fontsize=17)
cf2= reg_map2(ax, CMVHR_SLP_reg, domain, cmap, levs, transform_map)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 10)
ccrs_plot(ax, 140, 235, 35, 60)

cax1 = fig.add_axes([0.1, 0.05, 0.38, 0.01])
cb1  = fig.colorbar(cf1, cax=cax1, orientation='horizontal', ticks=np.arange(-0.6,0.61,0.06)[::5])
cb1.set_label(r'(K day$^{-1}$)', fontsize=13)

cax2 = fig.add_axes([0.59, 0.05, 0.38, 0.01])
cb2  = fig.colorbar(cf2, cax=cax2, orientation='horizontal', ticks=np.arange(-1.2,1.21,0.12)[::5])
cb2.set_label('(hPa)', fontsize=13)

plt.tight_layout(w_pad=3,h_pad=-3)
plt.show()
