import pickle
import numpy as np
import sacpy as scp
import pandas as pd
import xarray as xr

import re, os, sys, glob
sys.path.append('./library')
from cplot  import *
from cbasic import *

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

syear = '1950'; eyear = '2014'; region1 = [10, 60, 100, 200]
CELR_SSH  = data_read('CESM1-CAM5-SE', 'CESM1-CAM5-SE-LR', 'SSH', 'SSH',  region1).sel(time=slice(syear,eyear))
CEHR_SSH  = data_read('CESM1-CAM5-SE', 'CESM1-CAM5-SE-HR', 'SSH', 'SSH',  region1).sel(time=slice(syear,eyear))
CMHR_SSH  = data_read('CMCC-CM2',      'CMCC-CM2-HR4',     'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
CMVHR_SSH = data_read('CMCC-CM2',      'CMCC-CM2-VHR4',    'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
CNLR_SSH  = data_read('CNRM-CM6-1',    'CNRM-CM6-1',       'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
CNHR_SSH  = data_read('CNRM-CM6-1',    'CNRM-CM6-1-HR',    'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
ECLR_SSH  = data_read('EC-Earth3P',    'EC-Earth3P',       'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
ECHR_SSH  = data_read('EC-Earth3P',    'EC-Earth3P-HR',    'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
HGLL_SSH  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-LL',  'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
HGMM_SSH  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-MM',  'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
HGHM_SSH  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-HM',  'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
HGHH_SSH  = data_read('HadGEM3-GC31',  'HadGEM3-GC31-HH',  'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
MPHR_SSH  = data_read('MPI-ESM1-2',    'MPI-ESM1-2-HR',    'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
MPXR_SSH  = data_read('MPI-ESM1-2',    'MPI-ESM1-2-XR',    'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
EWLR_SSH  = data_read('ECMWF-IFS',     'ECMWF-IFS-LR',     'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
EWMR_SSH  = data_read('ECMWF-IFS',     'ECMWF-IFS-MR',     'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.
EWHR_SSH  = data_read('ECMWF-IFS',     'ECMWF-IFS-HR',     'SSH', 'zos',  region1).sel(time=slice(syear,eyear))*100.

def calc_KE_axis(data):
    data_KE  = data.sel(lat=slice(20,50), lon=slice(140,165)).mean(dim=['time','lon'])
    lat_KE   = np.array(data_KE.lat)
    lat_grad = (lat_KE[1:] + lat_KE[0:-1])/2.0
    KE_grad  = (data_KE[1:].values - data_KE[0:-1].values)/abs(lat_grad[0])
    KE_grad  = xr.DataArray(KE_grad, dims=['lat'], coords={'lat':lat_grad})
    KE_axis  = KE_grad.lat[KE_grad.argmin()].values
    return KE_axis

def KE_domain(KE_axis):
    x_KE = np.array([140.0, 165.0, 165.0, 140.0, 140.0])
    y_KE = np.array([KE_axis - 2., KE_axis - 2., KE_axis + 3., KE_axis + 3., KE_axis - 2.])
    return x_KE, y_KE

def calc_KE_index(name, data):
    _, anom_data, dtr_anom_data = calc_anomaly(data)
    axis = KE_axis[f'{name}']; x,y = KE_domain(axis)
    index = extract_index(dtr_anom_data, y.min(), y.max(), x.min(), x.max())
    return index

models = {'CNLR': CNLR_SSH, 'CNHR': CNHR_SSH, 'ECLR': ECLR_SSH, 'ECHR': ECHR_SSH,
          'MPHR': MPHR_SSH, 'MPXR': MPXR_SSH, 'CMHR': CMHR_SSH, 'CMVHR': CMVHR_SSH,
          'CELR': CELR_SSH, 'CEHR': CEHR_SSH,
          'HGLL': HGLL_SSH, 'HGMM': HGMM_SSH, 'HGHM': HGHM_SSH, 'HGHH': HGHH_SSH,
          'EWLR': EWLR_SSH, 'EWMR': EWMR_SSH, 'EWHR': EWHR_SSH}

KE_axis = {}; KE_index = {}
for model_name, ssh_data in models.items():
    KE_axis[model_name]  = calc_KE_axis(ssh_data)
    KE_index[model_name] = calc_KE_index(model_name, ssh_data)

###########################################################################################################
output = {
    "KE_axis":  KE_axis,
    "KE_index": KE_index}
output_dir = "../figure/"
output_file = output_dir+"KE_"+syear+"to"+eyear+".pkl"

if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "wb") as f:
    pickle.dump(output, f)
