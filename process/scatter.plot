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

syear = '1950'; eyear = '2014'; region = [-20, 80, 100, 360]
CELR_LHF  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-LR', 'LHF', 'LHFLX', region).sel(time=slice(syear, eyear))
CEHR_LHF  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-HR', 'LHF', 'LHFLX', region).sel(time=slice(syear, eyear))
CMHR_LHF  = data_read('CMCC-CM2',       'CMCC-CM2-HR4',     'LHF',  'hfls', region).sel(time=slice(syear, eyear))
CMVHR_LHF = data_read('CMCC-CM2',       'CMCC-CM2-VHR4',    'LHF',  'hfls', region).sel(time=slice(syear, eyear))
CNLR_LHF  = data_read('CNRM-CM6-1',     'CNRM-CM6-1',       'LHF',  'hfls', region).sel(time=slice(syear, eyear))
CNHR_LHF  = data_read('CNRM-CM6-1',     'CNRM-CM6-1-HR',    'LHF',  'hfls', region).sel(time=slice(syear, eyear))
ECLR_LHF  = data_read('EC-Earth3P',     'EC-Earth3P',       'LHF',  'hfls', region).sel(time=slice(syear, eyear))
ECHR_LHF  = data_read('EC-Earth3P',     'EC-Earth3P-HR',    'LHF',  'hfls', region).sel(time=slice(syear, eyear))
HGLL_LHF  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-LL',  'LHF',  'hfls', region).sel(time=slice(syear, eyear))
HGMM_LHF  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-MM',  'LHF',  'hfls', region).sel(time=slice(syear, eyear))
HGHM_LHF  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-HM',  'LHF',  'hfls', region).sel(time=slice(syear, eyear))
HGHH_LHF  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-HH',  'LHF',  'hfls', region).sel(time=slice(syear, eyear))
MPHR_LHF  = data_read('MPI-ESM1-2',     'MPI-ESM1-2-HR',    'LHF',  'hfls', region).sel(time=slice(syear, eyear))
MPXR_LHF  = data_read('MPI-ESM1-2',     'MPI-ESM1-2-XR',    'LHF',  'hfls', region).sel(time=slice(syear, eyear))
EWLR_LHF  = data_read('ECMWF-IFS',      'ECMWF-IFS-LR',     'LHF',  'hfls', region).sel(time=slice(syear, eyear))
EWMR_LHF  = data_read('ECMWF-IFS',      'ECMWF-IFS-MR',     'LHF',  'hfls', region).sel(time=slice(syear, eyear))
EWHR_LHF  = data_read('ECMWF-IFS',      'ECMWF-IFS-HR',     'LHF',  'hfls', region).sel(time=slice(syear, eyear))

CELR_SLP  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-LR', 'PSL', 'PSL',   region).sel(time=slice(syear, eyear))/100.
CEHR_SLP  = data_read('CESM1-CAM5-SE',  'CESM1-CAM5-SE-HR', 'PSL', 'PSL',   region).sel(time=slice(syear, eyear))/100.
CMHR_SLP  = data_read('CMCC-CM2',       'CMCC-CM2-HR4',     'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
CMVHR_SLP = data_read('CMCC-CM2',       'CMCC-CM2-VHR4',    'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
CNLR_SLP  = data_read('CNRM-CM6-1',     'CNRM-CM6-1',       'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
CNHR_SLP  = data_read('CNRM-CM6-1',     'CNRM-CM6-1-HR',    'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
ECLR_SLP  = data_read('EC-Earth3P',     'EC-Earth3P',       'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
ECHR_SLP  = data_read('EC-Earth3P',     'EC-Earth3P-HR',    'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
HGLL_SLP  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-LL',  'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
HGMM_SLP  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-MM',  'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
HGHM_SLP  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-HM',  'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
HGHH_SLP  = data_read('HadGEM3-GC31',   'HadGEM3-GC31-HH',  'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
MPHR_SLP  = data_read('MPI-ESM1-2',     'MPI-ESM1-2-HR',    'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
MPXR_SLP  = data_read('MPI-ESM1-2',     'MPI-ESM1-2-XR',    'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
EWLR_SLP  = data_read('ECMWF-IFS',      'ECMWF-IFS-LR',     'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
EWMR_SLP  = data_read('ECMWF-IFS',      'ECMWF-IFS-MR',     'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.
EWHR_SLP  = data_read('ECMWF-IFS',      'ECMWF-IFS-HR',     'PSL',  'psl',  region).sel(time=slice(syear, eyear))/100.

CELR_Q925 = data_read2('CESM1-CAM5-SE', 'CESM1-CAM5-SE-LR', 'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
CEHR_Q925 = data_read2('CESM1-CAM5-SE', 'CESM1-CAM5-SE-HR', 'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
CMHR_Q925 = data_read2('CMCC-CM2',      'CMCC-CM2-HR4',     'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
CMVHR_Q925= data_read2('CMCC-CM2',      'CMCC-CM2-VHR4',    'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
CNLR_Q925 = data_read2('CNRM-CM6-1',    'CNRM-CM6-1',       'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
CNHR_Q925 = data_read2('CNRM-CM6-1',    'CNRM-CM6-1-HR',    'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
ECLR_Q925 = data_read2('EC-Earth3P',    'EC-Earth3P',       'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
ECHR_Q925 = data_read2('EC-Earth3P',    'EC-Earth3P-HR',    'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
HGLL_Q925 = data_read2('HadGEM3-GC31',  'HadGEM3-GC31-LL',  'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
HGMM_Q925 = data_read2('HadGEM3-GC31',  'HadGEM3-GC31-MM',  'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
HGHM_Q925 = data_read2('HadGEM3-GC31',  'HadGEM3-GC31-HM',  'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
HGHH_Q925 = data_read2('HadGEM3-GC31',  'HadGEM3-GC31-HH',  'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
MPHR_Q925 = data_read2('MPI-ESM1-2',    'MPI-ESM1-2-HR',    'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
MPXR_Q925 = data_read2('MPI-ESM1-2',    'MPI-ESM1-2-XR',    'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
EWLR_Q925 = data_read2('ECMWF-IFS',     'ECMWF-IFS-LR',     'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
EWMR_Q925 = data_read2('ECMWF-IFS',     'ECMWF-IFS-MR',     'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()
EWHR_Q925 = data_read2('ECMWF-IFS',     'ECMWF-IFS-HR',     'Q925', 'Q1').sel(time=slice(syear, eyear)).squeeze()

with open("../figure/KE_1950to2014.pkl", "rb") as f:
    data = pickle.load(f)
KE_axis  = data['KE_axis']
KE_index = data['KE_index']

def calc_KOE_index(name, index_data, target_data):
    axis   = KE_axis[f'{name}']; x,y = KOE_domain(axis)
    index  = norm(season(index_data[f'{name}'].sel(time=slice(syear, eyear)), [9,10,11]))[:-1]
    _, _, dtr_anom_target = calc_anomaly(target_data.sel(time=slice(syear, eyear)))
    target = season(extract_index(dtr_anom_target, y.min(), y.max(), x.min(), x.max()), [12,1])
    lnreg  = scp.LinReg(index, target)
    return lnreg

def calc_NP_index(name, index_data, target_data):
    index  = norm(season(index_data[f'{name}'].sel(time=slice(syear, eyear)), [9,10,11]))[:-1]
    _, _, dtr_anom_target = calc_anomaly(target_data.sel(time=slice(syear, eyear)))
    target = season(extract_index(dtr_anom_target, 35, 60, 140, 235), [12,1])
    lnreg  = scp.LinReg(index, target)
    return lnreg

names = ['CNLR', 'CNHR', 'ECLR', 'ECHR', 'MPHR', 'MPXR', 'CMHR', 'CMVHR', 'CELR', 'CEHR', 'HGLL', 'HGMM', 'HGHM', 'HGHH', 'EWLR', 'EWMR', 'EWHR']
LHF_index = {name: {} for name in names}
LHF_index['CNLR']  = calc_KOE_index('CNLR',  KE_index,  CNLR_LHF)
LHF_index['CNHR']  = calc_KOE_index('CNHR',  KE_index,  CNHR_LHF)
LHF_index['ECLR']  = calc_KOE_index('ECLR',  KE_index,  ECLR_LHF)
LHF_index['ECHR']  = calc_KOE_index('ECHR',  KE_index,  ECHR_LHF)
LHF_index['MPHR']  = calc_KOE_index('MPHR',  KE_index,  MPHR_LHF)
LHF_index['MPXR']  = calc_KOE_index('MPXR',  KE_index,  MPXR_LHF)
LHF_index['CMHR']  = calc_KOE_index('CMHR',  KE_index,  CMHR_LHF)
LHF_index['CMVHR'] = calc_KOE_index('CMVHR', KE_index, CMVHR_LHF)
LHF_index['CELR']  = calc_KOE_index('CELR',  KE_index,  CELR_LHF)
LHF_index['CEHR']  = calc_KOE_index('CEHR',  KE_index,  CEHR_LHF)
LHF_index['HGLL']  = calc_KOE_index('HGLL',  KE_index,  HGLL_LHF)
LHF_index['HGMM']  = calc_KOE_index('HGMM',  KE_index,  HGMM_LHF)
LHF_index['HGHM']  = calc_KOE_index('HGHM',  KE_index,  HGHM_LHF)
LHF_index['HGHH']  = calc_KOE_index('HGHH',  KE_index,  HGHH_LHF)
LHF_index['EWLR']  = calc_KOE_index('EWLR',  KE_index,  EWLR_LHF)
LHF_index['EWMR']  = calc_KOE_index('EWMR',  KE_index,  EWMR_LHF)
LHF_index['EWHR']  = calc_KOE_index('EWHR',  KE_index,  EWHR_LHF)

SLP_index = {name: {} for name in names}
SLP_index['CNLR']  = calc_NP_index('CNLR',  KE_index,  CNLR_SLP)
SLP_index['CNHR']  = calc_NP_index('CNHR',  KE_index,  CNHR_SLP)
SLP_index['ECLR']  = calc_NP_index('ECLR',  KE_index,  ECLR_SLP)
SLP_index['ECHR']  = calc_NP_index('ECHR',  KE_index,  ECHR_SLP)
SLP_index['MPHR']  = calc_NP_index('MPHR',  KE_index,  MPHR_SLP)
SLP_index['MPXR']  = calc_NP_index('MPXR',  KE_index,  MPXR_SLP)
SLP_index['CMHR']  = calc_NP_index('CMHR',  KE_index,  CMHR_SLP)
SLP_index['CMVHR'] = calc_NP_index('CMVHR', KE_index,  CMVHR_SLP)
SLP_index['CELR']  = calc_NP_index('CELR',  KE_index,  CELR_SLP)
SLP_index['CEHR']  = calc_NP_index('CEHR',  KE_index,  CEHR_SLP)
SLP_index['HGLL']  = calc_NP_index('HGLL',  KE_index,  HGLL_SLP)
SLP_index['HGMM']  = calc_NP_index('HGMM',  KE_index,  HGMM_SLP)
SLP_index['HGHM']  = calc_NP_index('HGHM',  KE_index,  HGHM_SLP)
SLP_index['HGHH']  = calc_NP_index('HGHH',  KE_index,  HGHH_SLP)
SLP_index['EWLR']  = calc_NP_index('EWLR',  KE_index,  EWLR_SLP)
SLP_index['EWMR']  = calc_NP_index('EWMR',  KE_index,  EWMR_SLP)
SLP_index['EWHR']  = calc_NP_index('EWHR',  KE_index,  EWHR_SLP)

Q925_index = {name: {} for name in names}
Q925_index['CNLR']  = calc_KOE_index('CNLR',  KE_index,  CNLR_Q925)
Q925_index['CNHR']  = calc_KOE_index('CNHR',  KE_index,  CNHR_Q925)
Q925_index['ECLR']  = calc_KOE_index('ECLR',  KE_index,  ECLR_Q925)
Q925_index['ECHR']  = calc_KOE_index('ECHR',  KE_index,  ECHR_Q925)
Q925_index['MPHR']  = calc_KOE_index('MPHR',  KE_index,  MPHR_Q925)
Q925_index['MPXR']  = calc_KOE_index('MPXR',  KE_index,  MPXR_Q925)
Q925_index['CMHR']  = calc_KOE_index('CMHR',  KE_index,  CMHR_Q925)
Q925_index['CMVHR'] = calc_KOE_index('CMVHR', KE_index,  CMVHR_Q925)
Q925_index['CELR']  = calc_KOE_index('CELR',  KE_index,  CELR_Q925)
Q925_index['CEHR']  = calc_KOE_index('CEHR',  KE_index,  CEHR_Q925)
Q925_index['HGLL']  = calc_KOE_index('HGLL',  KE_index,  HGLL_Q925)
Q925_index['HGMM']  = calc_KOE_index('HGMM',  KE_index,  HGMM_Q925)
Q925_index['HGHM']  = calc_KOE_index('HGHM',  KE_index,  HGHM_Q925)
Q925_index['HGHH']  = calc_KOE_index('HGHH',  KE_index,  HGHH_Q925)
Q925_index['EWLR']  = calc_KOE_index('EWLR',  KE_index,  EWLR_Q925)
Q925_index['EWMR']  = calc_KOE_index('EWMR',  KE_index,  EWMR_Q925)
Q925_index['EWHR']  = calc_KOE_index('EWHR',  KE_index,  EWHR_Q925)

output = {"LHF_index": LHF_index}
output_dir = "../figure/"
output_file = output_dir+"scatter_LHF_"+syear+"to"+eyear+".pkl"
if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "wb") as f:
    pickle.dump(output, f)

output = {"Q925_index": Q925_index}
output_dir = "../figure/"
output_file = output_dir+"scatter_Q925_"+syear+"to"+eyear+".pkl"
if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "wb") as f:
    pickle.dump(output, f)

output = {"SLP_index": SLP_index}
output_dir = "../figure/"
output_file = output_dir+"scatter_SLP_"+syear+"to"+eyear+".pkl"
if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, "wb") as f:
    pickle.dump(output, f)
    
