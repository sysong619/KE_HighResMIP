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

syear = '1993'; eyear = '2022'; region1 = [18, 62, -180, 180]; region2 = [-20, 80, 100, 300]
SSH = data_read('OBS', 'AVISO', 'SSH', 'adt', region1).sel(time=slice(syear,eyear)); SSH = lonflip(SSH, 2)
SST = data_read('OBS', 'OISST', 'SST', 'sst', region2).sel(time=slice(syear,eyear))
LHF = data_read('OBS', 'J-OFURO3', 'LHF', 'LHF', region2).sel(time=slice(syear,eyear))
SHF = data_read('OBS', 'J-OFURO3', 'SHF', 'SHF', region2).sel(time=slice(syear,eyear))
THF = SHF + LHF
MSK = data_read('REANAL', 'ERA5',    'SST', 'sst', region2).sel(time=slice(syear,eyear))-273.15
SLP = data_read('REANAL', 'ERA5',    'SLP', 'msl', region2).sel(time=slice(syear,eyear))/100.
UWD = data_read('REANAL', 'ERA5',    'UWD', 'u',   region2).sel(time=slice(syear,eyear)).sel(level=slice(925,925)).squeeze()
VWD = data_read('REANAL', 'ERA5',    'VWD', 'v',   region2).sel(time=slice(syear,eyear)).sel(level=slice(925,925)).squeeze()
DBH = data_read('REANAL', 'ERA5',    'DBH', 'q1',  region2).sel(time=slice(syear,eyear)).sel(level=slice(925,925)).squeeze()
DBH = DBH.where(~np.isnan(MSK))

_, _, dtr_anom_SSH = calc_anomaly(SSH)
_, _, dtr_anom_SST = calc_anomaly(SST)
_, _, dtr_anom_THF = calc_anomaly(THF)
_, _, dtr_anom_SLP = calc_anomaly(SLP)
_, _, dtr_anom_UWD = calc_anomaly(UWD)
_, _, dtr_anom_VWD = calc_anomaly(VWD)
_, _, dtr_anom_DBH = calc_anomaly(DBH)

index   = norm(season(extract_index(dtr_anom_SSH, 31, 36, 140, 165), [9,10,11]))[:-1]
target  = season(dtr_anom_THF, [12,1]); THF_reg = scp.LinReg(index, target)
target  = season(dtr_anom_SST, [12,1]); SST_reg = scp.LinReg(index, target)
target  = season(dtr_anom_DBH, [12,1]); DBH_reg = scp.LinReg(index, target)
target  = season(dtr_anom_SLP, [12,1]); SLP_reg = scp.LinReg(index, target)
target  = season(dtr_anom_UWD, [12,1]); UWD_reg = scp.LinReg(index, target)
target  = season(dtr_anom_VWD, [12,1]); VWD_reg = scp.LinReg(index, target)

fig = plt.figure(figsize=(10,8)); gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1,1], width_ratios=[1,1])
projection_map = ccrs.PlateCarree(central_longitude=180); transform_map = ccrs.PlateCarree()
domain = [18, 62, 110, 250]

cmap = cmap_white_center(plt.cm.RdBu_r); levs = np.arange(-48,48.1,4.8)
ax = plt.subplot(gs[0,0], projection=projection_map); ax.set_title('a', weight='bold', loc='left', fontsize=20)
cb = reg_map(ax, THF_reg, domain, cmap, levs, transform_map); cb.set_label(r'(W m$^{-2}$)', fontsize=13)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 13); ccrs_plot(ax, 140, 175, 31, 45)

cmap = cmap_white_center(plt.cm.RdBu_r); levs = np.arange(-0.6,0.61,0.06)
ax = plt.subplot(gs[0,1], projection=projection_map); ax.set_title('b', weight='bold', loc='left', fontsize=20)
cb = reg_map(ax, SST_reg, domain, cmap, levs, transform_map); cb.set_label('(°C)', fontsize=13)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 13); ccrs_plot(ax, 140, 175, 31, 45)

cmap = cmap_white_center(plt.cm.PiYG_r); levs = np.arange(-0.6,0.61,0.06)
ax = plt.subplot(gs[1,0], projection=projection_map); ax.set_title('c', weight='bold', loc='left', fontsize=20)
cb = reg_map(ax, DBH_reg, domain, cmap, levs, transform_map); cb.set_label(r'(K day$^{-1}$)', fontsize=13)
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 13); ccrs_plot(ax, 140, 175, 31, 45)

cmap = cmap_white_center(plt.cm.coolwarm); levs = np.arange(-1.2,1.21,0.12)
ax = plt.subplot(gs[1,1], projection=projection_map); ax.set_title('d', weight='bold', loc='left', fontsize=20)
cb = reg_map(ax, SLP_reg, domain, cmap, levs, transform_map); cb.set_label('(hPa)', fontsize=13)
qv = quiver_reg(ax, UWD_reg, VWD_reg, domain, transform_map, siglev=0.1)
ax.quiverkey(qv, X=0.85, Y=1.05, U=1, label=r'1 m s$^{-1}$', labelpos='E')
ccrs_grid(ax, np.arange(120,240.1,30), np.arange(20,60.1,10), 13); ccrs_plot(ax, 140, 235, 35, 60)

plt.tight_layout(w_pad=3, h_pad=2)
plt.show()
