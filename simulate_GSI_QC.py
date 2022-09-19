# Dependencies (PyGSI compliant)
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import time
from simulate_GSI_QC_dependencies import prRed1
from simulate_GSI_QC_dependencies import prRed2
from simulate_GSI_QC_dependencies import prGreen1
from simulate_GSI_QC_dependencies import prGreen2
from simulate_GSI_QC_dependencies import GSI_JEDI_QC_contingency_table
from simulate_GSI_QC_dependencies import uwdvwd_to_spddir
from simulate_GSI_QC_dependencies import search_flags
from simulate_GSI_QC_dependencies import convinfo
from simulate_GSI_QC_dependencies import p2lev
from simulate_GSI_QC_dependencies import simulate_GSI_QC_satwinds

# Extract data from diag, geovals files
diag_file='satwind_diag_2021080100_0000.nc4'
geov_file='satwind_geoval_2021080100.nc4'
diag_hdl=Dataset(diag_file)
geov_hdl=Dataset(geov_file)
ob_u=np.asarray(diag_hdl['ObsValue']['eastward_wind']).squeeze()
ob_v=np.asarray(diag_hdl['ObsValue']['northward_wind']).squeeze()
ufo_u=np.asarray(diag_hdl['hofx']['eastward_wind']).squeeze()
ufo_v=np.asarray(diag_hdl['hofx']['northward_wind']).squeeze()
gsi_u=np.asarray(diag_hdl['GsiHofX']['eastward_wind']).squeeze()
gsi_v=np.asarray(diag_hdl['GsiHofX']['northward_wind']).squeeze()
gsi_err1=np.asarray(diag_hdl['GsiInputObsError']['eastward_wind'])
gsi_err2=np.asarray(diag_hdl['GsiAdjustObsError']['eastward_wind'])
gsi_err3=np.asarray(diag_hdl['GsiFinalObsError']['eastward_wind'])
ob_lat=np.asarray(diag_hdl['MetaData']['latitude']).squeeze()
ob_lon=np.asarray(diag_hdl['MetaData']['longitude']).squeeze()
ob_pre=np.asarray(diag_hdl['MetaData']['air_pressure']).squeeze()
ob_typu=np.asarray(diag_hdl['ObsType']['eastward_wind']).squeeze()
ob_typv=np.asarray(diag_hdl['ObsType']['northward_wind']).squeeze()
ufo_qc_u=np.asarray(diag_hdl['EffectiveQC']['eastward_wind']).squeeze()
ufo_qc_v=np.asarray(diag_hdl['EffectiveQC']['northward_wind']).squeeze()
gsi_qc_u=np.asarray(diag_hdl['GsiEffectiveQC']['eastward_wind']).squeeze()
gsi_qc_v=np.asarray(diag_hdl['GsiEffectiveQC']['northward_wind']).squeeze()
gsi_qm_u=np.asarray(diag_hdl['PreQC']['eastward_wind']) # quality-mark, qm in code
gsi_qm_v=np.asarray(diag_hdl['PreQC']['northward_wind']) # quality-mark, qm in code
ob_err_u = np.asarray(diag_hdl['ObsError']['eastward_wind']).squeeze()
ob_err_v = np.asarray(diag_hdl['ObsError']['northward_wind']).squeeze()
# geovals
geo_lat=np.asarray(geov_hdl.variables['latitude']).squeeze()
geo_lon=np.asarray(geov_hdl.variables['longitude']).squeeze()
geo_u=np.asarray(geov_hdl.variables['eastward_wind']).squeeze()
geo_v=np.asarray(geov_hdl.variables['northward_wind']).squeeze()
geo_pre=np.asarray(geov_hdl.variables['air_pressure']).squeeze()
geo_lev=np.asarray(geov_hdl.variables['air_pressure_levels']).squeeze()
geo_ps=np.asarray(geov_hdl.variables['surface_pressure']).squeeze()
geo_tp=np.asarray(geov_hdl.variables['tropopause_pressure']).squeeze()
geo_isli=np.asarray(geov_hdl.variables['land_type_index_NPOESS']).squeeze()
# Compute wind speed, direction
ufo_spd,ufo_dir=uwdvwd_to_spddir(ufo_u,ufo_v)
gsi_spd,gsi_dir=uwdvwd_to_spddir(gsi_u,gsi_v)
ob_spd,ob_dir=uwdvwd_to_spddir(ob_u,ob_v)

# Run Test
t0=time.time()
sim_qc_logic,sim_qc_flag=simulate_GSI_QC_satwinds(gsi_err1,gsi_err2,ob_typu,gsi_qm_u,ob_pre,geo_ps,geo_lev,
                                     geo_tp,geo_isli,ob_spd,ufo_spd,ob_dir,ufo_dir,ob_u-ufo_u,ob_v-ufo_v,convinfo)
t1=time.time()
sim_qc_u=np.ones(np.shape(sim_qc_logic),dtype='int32')
sim_qc_u[np.where(sim_qc_logic)]=0
sim_qc_flag=np.asarray(sim_qc_flag,dtype='object')
print('completed in {:.2f} seconds'.format(t1-t0))

# Confirm Test
print(np.corrcoef(gsi_qc_u,sim_qc_u))

