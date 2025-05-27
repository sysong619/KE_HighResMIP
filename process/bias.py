import pickle
import numpy as np
import sacpy as scp
import pandas as pd
import xarray as xr

import re, os, sys, glob
sys.path.append('./library')
from cplot  import *
from cbasic import *

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

syear = '1950'; eyear = '2014'; region = [-90, 90, 0, 360]
SST   = data_read('REANAL', 'ERA5',  'SST', 'sst',      region).sel(time=slice(syear,eyear))-273.15
target_lat = SST.lat; target_lon = SST.lon

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
    data = ds[f"{varname}"].sortby('lat').sel(lat=slice(latS, latN), lon=slice(lonW, lonE)).interp(lat=target_lat, lon=target_lon)
    return data

CELR_SST  = data_read('CESM1-CAM5-SE', 'CESM1-CAM5-SE-LR', 'SST', 'TEMP', region).sel(time=slice(syear, eyear))
CEHR_SST  = data_read('CESM1-CAM5-SE', 'CESM1-CAM5-SE-HR', 'SST', 'TEMP', region).sel(time=slice(syear, eyear))
CMHR_SST  = data_read('CMCC-CM2',      'CMCC-CM2-HR4',     'SST', 'tos',  region).sel(time=slice(syear, eyear))
CMVHR_SST = data_read('CMCC-CM2',      'CMCC-CM2-VHR4',    'SST', 'tos',  region).sel(time=slice(syear, eyear))
CNLR_SST  = data_read('CNRM-CM6-1',    'CNRM-CM6-1',       'SST', 'tos',  region).sel(time=slice(syear, eyear))
CNHR_SST  = data_read('CNRM-CM6-1',    'CNRM-CM6-1-HR',    'SST', 'tos',  region).sel(time=slice(syear, eyear))
ECLR_SST  = data_read('EC-Earth3P',    'EC-Earth3P',       'SST', 'tos',  region).sel(time=slice(syear, eyear))
ECHR_SST  = data_read('EC-Earth3P',    'EC-Earth3P-HR',    'SST', 'tos',  region).sel(time=slice(syear, eyear))
HGLL_SST  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-LL',  'SST', 'tos',  region).sel(time=slice(syear, eyear))
HGMM_SST  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-MM',  'SST', 'tos',  region).sel(time=slice(syear, eyear))
HGHM_SST  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-HM',  'SST', 'tos',  region).sel(time=slice(syear, eyear))
HGHH_SST  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-HH',  'SST', 'tos',  region).sel(time=slice(syear, eyear))
MPHR_SST  = data_read('MPI-ESM1-2',    'MPI-ESM1-2-HR',    'SST', 'tos',  region).sel(time=slice(syear, eyear))
MPXR_SST  = data_read('MPI-ESM1-2',    'MPI-ESM1-2-XR',    'SST', 'tos',  region).sel(time=slice(syear, eyear))
EWLR_SST  = data_read('ECMWF-IFS',     'ECMWF-IFS-LR',     'SST', 'tos',  region).sel(time=slice(syear, eyear))
EWMR_SST  = data_read('ECMWF-IFS',     'ECMWF-IFS-MR',     'SST', 'tos',  region).sel(time=slice(syear, eyear))
EWHR_SST  = data_read('ECMWF-IFS',     'ECMWF-IFS-HR',     'SST', 'tos',  region).sel(time=slice(syear, eyear))

with open("./KE_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
KE_axis = data['KE_axis']; KE_index = data['KE_index']

def KOE_SST_DJ_bias(name, CMIP_SST, SST, syear=syear, eyear=eyear):
    time = pd.date_range(start=f'{syear}-01-01', end=f'{eyear}-12-31', freq='M')
    CMIP_SST['time'] = time; axis  = KE_axis[f'{name}']; x, y = KOE_domain(axis)
    SST_bias = CMIP_SST.sel(time=slice(syear,eyear)) - SST.sel(time=slice(syear,eyear))
    KOE_SST_bias = extract_index(SST_bias, y.min(), y.max(), x.min(), x.max())
    return season(KOE_SST_bias, [12,1]).mean(axis=0)

names = ['CNLR', 'CNHR', 'ECLR', 'ECHR', 'MPHR', 'MPXR', 'CMHR', 'CMVHR', 'CELR', 'CEHR', 'HGLL', 'HGMM', 'HGHM', 'HGHH', 'EWLR', 'EWMR', 'EWHR']
SST_vars = [CNLR_SST, CNHR_SST, ECLR_SST, ECHR_SST, MPHR_SST, MPXR_SST, CMHR_SST, CMVHR_SST,
            CELR_SST, CEHR_SST, HGLL_SST, HGMM_SST, HGHM_SST, HGHH_SST, EWLR_SST, EWMR_SST, EWHR_SST]
SST_bias_index = {}
for name, sst_cmip in zip(names, SST_vars):
    SST_bias_index[name] = KOE_SST_DJ_bias(name, sst_cmip, SST)

output = {"SST_bias_index": SST_bias_index}
output_dir = "../figure/"
output_file = output_dir+"SST_bias_"+syear+"to"+eyear+".pkl"

if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "wb") as f:
    pickle.dump(output, f)
