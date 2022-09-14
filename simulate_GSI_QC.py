# Dependencies (PyGSI compliant)
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Functions
# Color-coded text outputs
# 1 output string
def prRed1(skk1): print("\033[91m {}\033[00m" .format(skk1))
def prGreen1(skk1): print("\033[92m {}\033[00m" .format(skk1))
# 2 output strings
def prRed2(skk1,skk2): print("\033[91m {}\033[00m" .format(skk1),"\033[91m {}\033[00m" .format(skk2))
def prGreen2(skk1,skk2): print("\033[92m {}\033[00m" .format(skk1),"\033[92m {}\033[00m" .format(skk2))
        
def GSI_JEDI_QC_contingency_table(GSI_qc,JEDI_qc,ObsType,jediFlag=None,reportTypes=False):
    # Report contingency table for GSI_qc and JEDI_qc:
    #   Percent of obs both GSI and JEDI pass (associated Ob-Types)
    #   Percent of obs GSI pass and JEDI fail (associated Ob-Types)
    #   Percent of obs GSI fail and JEID pass (associated Ob-Types)
    #   Percent of obs both GSI and JEDI fail (associated Ob-Types)
    # The jediFlag can be set to specify a particular subset of JEDI
    # QC failure based on the flag number. A value of jediFlag=None
    # (default) will use any flag value (JEDI_qc != 0)
    # Inputs:
    #   GSI_qc: gsiEffectiveQC values for subset of obs being tested
    #   JEDI_QC: effectiveQC values for subset of obs being tested
    #   ObsType: Ob-Types for subset of obs being tested
    #   jediFlag: optional value or list of values to specify JEDI rejection type
    #   reportTypes: optional value to report all types for each contingency
    if jediFlag==None:
        Gqc = GSI_qc
        Jqc = JEDI_qc
        total_obs=np.size(Jqc)
    elif type(jediFlag) is not list:
        jediFlag = [jediFlag]
        idx = np.where(np.isin(JEDI_qc,jediFlag))
        Gqc = GSI_qc[idx]
        Jqc = JEDI_qc[idx]
        total_obs=np.size(Jqc)
    else:
        idx = np.where(np.isin(JEDI_qc,jediFlag))
        Gqc = GSI_qc[idx]
        Jqc = JEDI_qc[idx]
        total_obs=np.size(Jqc)
    print('    Total Obs: {:d}'.format(total_obs))
    if total_obs>0:
        # Report contingency table, with or without types
        # Both GSI and JEDI pass
        idx=np.where((Gqc==0)&(Jqc==0))
        table_obs=np.size(idx)
        type_obs=np.unique(ObsType[idx])
        if reportTypes:
            prGreen2('    Flagged obs pass both GSI and JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs)+' Types:',type_obs)
        else:
            prGreen1('    Flagged obs pass both GSI and JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs))
        # GSI fails, JEDI passes    
        idx=np.where((Gqc!=0)&(Jqc==0))
        table_obs=np.size(idx)
        type_obs=np.unique(ObsType[idx])
        if reportTypes:
            prRed2('    Flagged obs fail GSI and pass JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs)+' Types:',type_obs)
        else:
            prRed1('    Flagged obs fail GSI and pass JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs))
        # GSI passes, JEDI fails
        idx=np.where((Gqc==0)&(Jqc!=0))
        table_obs=np.size(idx)
        type_obs=np.unique(ObsType[idx])
        if reportTypes:
            prRed2('    Flagged obs pass GSI and fail JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs)+' Types:',type_obs)
        else:
            prRed1('    Flagged obs pass GSI and fail JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs))
        # Both GSI and JEDI fail
        idx=np.where((Gqc!=0)&(Jqc!=0))
        table_obs=np.size(idx)
        type_obs=np.unique(ObsType[idx])
        if reportTypes:
            prGreen2('    Flagged obs fail both GSI and JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs)+' Types:',type_obs)
        else:
            prGreen1('    Flagged obs fail both GSI and JEDI QC {:.2f}% ({:d})'.format(100.*table_obs/total_obs,table_obs))
    else:
        # No contingency table can be produced for this subset
        print('    No contingency table for this subset')
    return


def uwdvwd_to_spddir(uwd,vwd):
    spd=np.sqrt(uwd**2.+vwd**2.)
    ang=(270.-np.arctan2(vwd,uwd)*(180./np.pi))%(360.)
    return spd, ang

def wdir_diff(dir1,dir2):
    ndiff = np.size(dir1)
    diff1 = np.abs(dir2-dir1)
    diff2 = np.abs(dir2-dir1+360.)
    diff3 = np.abs(dir2-dir1-360.)
    diff = np.nan*np.ones(ndiff,)
    for i in range(ndiff):
        diff[i]=np.min(np.asarray([diff1[i],diff2[i],diff3[i]]).squeeze())
    return diff

def spdb_check(uo,vo,um,vm,e,emin,emax,thresh):
    residual=np.sqrt((uo-um)**2. + (vo-vm)**2.)
    e[e<emin]=emin
    e[e>emax]=emax
    return np.divide(residual,e)<thresh

def p2lev(po,pl,flg):
    # Recreates grdcrd1 and isrchf functions to find grid coordinate for pressure po given profile pl of size (nl,)
    # flg: 1 (inreasing p with index), -1 (decreasing)
    nl=np.size(pl)
    pdif=po-pl
    z = np.nan
    zdif=9.99e+10
    if flg == 1:
        for k in range(nl):
            if (np.abs(pdif[k])<np.abs(zdif))&(pdif[k]>=0):
                z=k
                zdif=pdif[k]
        if(po<=pl[0]):
            kp=0
        else:
            kp=z
    elif flg == -1:
        for k in range(nl):
            if (np.abs(pdif[k])<np.abs(zdif))&(pdif[k]<=0):
                z=k
                zdif=pdif[k]
        if(po>=pl[0]):
            kp=0
        else:
            kp=z
    d=float(kp)+(po-pl[kp])/(pl[kp+1]-pl[kp])
    return d

#          typ    cgross   ermin  ermax
convinfo={
           240 : [ 2.5,     6.1,   1.4],
           242 : [ 2.5,    15.0,   1.4],
           243 : [ 1.5,    15.0,   1.4],
           244 : [ 2.5,    20.0,   1.4],
           245 : [ 1.3,    20.0,   1.4],
           246 : [ 1.3,    20.0,   1.4],
           247 : [ 2.5,    20.0,   1.4],
           250 : [ 2.5,    20.0,   1.4],
           252 : [ 2.5,    20.0,   1.4],
           253 : [ 1.5,    20.0,   1.4],
           254 : [ 1.5,    20.0,   1.4],
           255 : [ 2.5,    20.1,   1.4],
           257 : [ 2.5,    20.1,   1.4],
           258 : [ 2.5,    20.1,   1.4],
           259 : [ 2.5,    20.1,   1.4],
           260 : [ 2.5,    20.1,   1.4],
         }

def p2lev(po,pl,flg):
    # Recreates grdcrd1 and isrchf functions to find grid coordinate for pressure po given profile pl of size (nl,)
    # flg: 1 (inreasing p with index), -1 (decreasing)
    nl=np.size(pl)
    pdif=po-pl
    z = np.nan
    zdif=9.99e+10
    if flg == 1:
        for k in range(nl):
            if (np.abs(pdif[k])<np.abs(zdif))&(pdif[k]>=0):
                z=k
                zdif=pdif[k]
        if(po<=pl[0]):
            kp=0
        else:
            kp=z
    elif flg == -1:
        for k in range(nl):
            if (np.abs(pdif[k])<np.abs(zdif))&(pdif[k]<=0):
                z=k
                zdif=pdif[k]
        if(po>=pl[0]):
            kp=0
        else:
            kp=z
    d=float(kp)+(po-pl[kp])/(pl[kp+1]-pl[kp])
    return d

def simulate_GSI_QC_satwinds(input_err,adjust_err,ob_typ,ob_qm,ob_p,ob_ps,ob_p_prof,
                             ob_tp,ob_isli,ob_speed,bk_speed,ob_direc,bk_direc,ob_u_omf,ob_v_omf,convlib):
    # Simulates setupw.f90 qcgross check
    #
    # Inputs:
    #  input_err: Input error value (from data(ier2,i))
    #  adjust_err: Adjusted error value (from data(ier,i))
    #  ob_typ: Observation type (from ob_typu or ob_typv)
    #  ob_qm: Observation quality-mark (from gsi_qm_u or gsi_qm_v)
    #  ob_p: Observation pressure (from data(ipres,i))
    #  ob_p_prof: Profile of pressure on sigma levels at observation lcation (derived from geovals)
    #  ob_ps: Model surface pressure at observation location (derived from geovals)
    #  ob_tp: Model tropopause pressure at observation location (derived from geovals)
    #  ob_isli: Model surface type (0=water,!0=land or ice) (derived from geovals) (? NOT IMPLEMENTED YET)
    #  ob_speed: Observation speed (derived from ob u/v)
    #  bk_speed: Model background speed (derived from ges u/v)
    #  ob_direc: Observation direction (derived from ob u/v)
    #  bk_direc: Model background direction (derived from ges u/v)
    #  ob_u_omf: Observation OmF zonal wind (derived from ob and ges u)
    #  ob_v_omf: Observation OmF merid wind (derived from ob and ges v)
    #  convlib: convinfo library containing list of [cgross,ermin,emax] for each ob type
    #
    # Outputs:
    #  qc_test: Results of GSI QC test (True==passes, False==fails)
    #  qc_flag: List of numerical flags based on what criteria tripped QC test (0==never tripped)
    #
    # qc_flag table:
    #   0: GSI QC test passes (qc_test==True)
    #   1: (ob pressure above model top), ratio_errors set to 0
    #   2: (ob pressure more than 50 hPa above tropopause), error set to 0
    #   3: (ob pressure more than 950 hPa), error set to 0
    #   4: (Type [242,243] ob pressure less than 700 hPa), error set to 0
    #   5: (Type [245] ob between 399-801 hPa), error set to 0
    #   6: (Type [252] ob between 499-801 hPa), error set to 0
    #   7: (Type [253] ob between 401-801 hPa), error set to 0
    #   8: (Type [257] ob pressure less than 249 hPa), error set to 0
    #   9: (Type [246,250,254] ob pressure greater than 399 hPa), error set to 0
    #  10: (Type [258] ob pressure greater than 600 hPa), error set to 0
    #  11: (Type [259] ob pressure greater than 600 hPa), error set to 0
    #  12: (Type [259] ob pressure less than 249 hPa), error set to 0
    #  13: (Type [247] ob fails LNVD check), error set to 0
    #  14: (Type [247] ob fails wdirdiff check), error set to 0
    #  15: (Type [257,258,259,260] ob fails LNVD check), error set to 0
    #  16: (Type [244] ob fails LNVD check), error set to 0
    #  17: (ob fails qcgross test), ratio_errors set to 0
    #  18: (ratio_errors*error falls beneath tiny_r_kind), qc_test assigned False
    #  19: (ob_qm is [9,12,15]), qc_test assigned False
    #  NOTE: 18 but not 17: Likely error is negative from input
    #
    # Line numbers for read_satwnd.f90 and setupw.f90 provided at end of each operation, some operations are
    # skipped, these are provided in comments only
    #
    # Initialize output test as None, flag as empty
    qc_test = None
    qc_flag = np.asarray([],dtype=np.int32)
    # Include hard-wired rsigp (number of sigma levels, plus 1)
    rsigp=128. # inferred from gridmod.F90
    rsig=rsigp-1 # inferred from L 397
    # Define cgross, ermin, and ermax from convlib based on ob_typ
    cgross,ermax,ermin=convlib[ob_typ]
    # setupw.f90:
    error = input_err # L 511
    obserror=max(ermin,min(ermax,adjust_err)) # L 579
    dpres = np.log(0.01*ob_p) # L 795
    dpres = dpres-np.log(0.01*ob_ps) # L 797
    drpx = 0. # L 798
    dpres=np.log(np.exp(dpres)*0.01*ob_ps) # L 805
    dpres=p2lev(dpres,np.log(0.01*ob_p_prof),-1)
    sfcchk=p2lev(np.log(0.01*ob_ps),np.log(0.01*ob_p_prof),-1) # L 868
    rlow=max(sfcchk-dpres,0.) # L 875
    rhgh=max(dpres-0.001-rsigp,0.) # L 876
    ratio_errors=error/(adjust_err+drpx+1.0e6*rhgh+4.*rlow) # L 882
    
    spdb=ob_speed-bk_speed # L 898
    
    error = 1./error # L 913
    
    if dpres>rsig:
        ratio_errors=0. # L 915-923
        qc_flag=np.append(qc_flag,1)
    if ob_p<ob_tp-5000.:
        error=0. # L 947-950
        qc_flag=np.append(qc_flag,2)
    if ob_p>95000.:
        error=0. # L 952-959
        qc_flag=np.append(qc_flag,3)
    
    if (np.isin(ob_typ,[242,243]))&(ob_p<70000.):
        error=0. # L 960-962
        qc_flag=np.append(qc_flag,4)
    if (np.isin(ob_typ,[245]))&(ob_p<80100.)&(ob_p>39900.):
        error=0. # L 963-967
        qc_flag=np.append(qc_flag,5)
    if (np.isin(ob_typ,[252]))&(ob_p<80100.)&(ob_p>49900.):
        error=0. # L 968-970
        qc_flag=np.append(qc_flag,6)
    if (np.isin(ob_typ,[253]))&(ob_p<80100.)&(ob_p>40100.):
        error=0. # L 971-975
        qc_flag=np.append(qc_flag,7)
    if (np.isin(ob_typ,[246,250,254]))&(ob_p>39900.):
        error=0. # L 976-978
        qc_flag=np.append(qc_flag,8)
    if (np.isin(ob_typ,[257]))&(ob_p<24900.):
        error=0. # L 979
        qc_flag=np.append(qc_flag,9)
    if (np.isin(ob_typ,[258]))&(ob_p>60000.):
        error=0. # L 980
        qc_flag=np.append(qc_flag,10)
    if (np.isin(ob_typ,[259]))&(ob_p>60000.):
        error=0. # L 981
        qc_flag=np.append(qc_flag,11)
    if (np.isin(ob_typ,[259]))&(ob_p<24900.):
        error=0. # L 982
        qc_flag=np.append(qc_flag,12)
    
    # Type 247 LNVD check, L 993-1002
    if (np.isin(ob_typ,[247])):
        if((np.sqrt(ob_u_omf**2.+ob_v_omf**2.)/np.log(ob_speed) >= 3.) |
           ((ob_p > ob_ps-11000.)&(ob_isli != 0))):
            error = 0.
            qc_flag=np.append(qc_flag,13)
    # Type 247 wind direction check, L 1004-1010
    if (np.isin(ob_typ,[247])):
        if (np.min([np.abs(ob_direc-bk_direc),np.abs(ob_direc-bk_direc+360.),np.abs(ob_direc-bk_direc-360.)])>50.):
            error = 0.
            qc_flag=np.append(qc_flag,14)
    # MODIS LNVD check, L 1022-1030
    if (np.isin(ob_typ,[257,258,259,260])):
        if((np.sqrt(ob_u_omf**2.+ob_v_omf**2.)/np.log(ob_speed) >= 3.) |
           ((ob_p > ob_ps-20000.)&(ob_isli != 0))):
            error = 0.
            qc_flag=np.append(qc_flag,15)
    # Type 244 LNVD check, L 1039-1048
    if (np.isin(ob_typ,[244])):
        if((np.sqrt(ob_u_omf**2.+ob_v_omf**2.)/np.log(ob_speed) >= 3.) |
           ((ob_p > ob_ps-20000.)&(ob_isli != 0))):
            error = 0.
            qc_flag=np.append(qc_flag,16)
    
    # Compute components of qcgross test
    obserror = 1./max(ratio_errors*error,1.0e-10) # L 1142
    obserrlm = max(ermin,min(ermax,obserror)) # L 1143
    residual = np.sqrt((ob_u_omf)**2+(ob_v_omf)**2) # L 1144
    ratio    = residual/obserrlm # L 1145
    qcgross=cgross # L 1148
    # qcgross adjustments
    if (ob_qm==3.): qcgross=0.7*cgross # L 1149-1151
    if (spdb<0.)&(np.isin(ob_typ,[244])): qcgross=0.7*cgross # L 1154-1156
    if (spdb<0.)&(np.isin(ob_typ,[245,246]))&(ob_p<40000.)&(ob_p>30000.): qcgross=0.7*cgross # L 1157-1159
    if (spdb<0.)&(np.isin(ob_typ,[253,254]))&(ob_p<40000.)&(ob_p>20000.): qcgross=0.7*cgross # L 1160-1162
    if (spdb<0.)&(np.isin(ob_typ,[257,258,259])): qcgross=0.7*cgross # L 1163-1165
    
    # qcgross test, L 1168-1174
    if (ratio>qcgross)|(ratio<1.0e-06):
        ratio_errors = 0.
        qc_flag=np.append(qc_flag,17)
    # else, ratio_errors is then divided by sqrt(dup)
    
    # muse set to false if ratio_errors*error<=tiny_r_kind, L 1206
    if ratio_errors*error<=1.0e-06:
        qc_test=False
        qc_flag=np.append(qc_flag,18)  # 18 but not 17: Likely error is negative from input
    else:
        qc_test=True
    
    # ratio_errors is then multiplied by sqrt(hilb), L 1232
    
    # error_final defined by ratio_errors*errors, or zero if ratio_errors*errors<=tiny_r_kind, L 1381-1385
    
    # read_satwnd.f90:
    if (np.isin(ob_qm,[9,12,15])):
        qc_test=False
        qc_flag=np.append(qc_flag,19)
    
    # Set qc_flag to contain 0 if passed
    if qc_test:
        qc_flag=np.asarray([0])
    return qc_test,qc_flag

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

ufo_dir_omf=wdir_diff(ob_dir,ufo_dir)
gsi_dir_omf=wdir_diff(ob_dir,gsi_dir)

n=150000
sim_qc_u=np.nan*np.ones((n,))
sim_qc_flg=[]
np.random.seed(90210)
sample=np.random.choice(np.arange(np.size(gsi_qc_u)),n,replace=False)
np.random.seed(None)
perc_complete=0.
print('{:.1f}% complete'.format(perc_complete))
for j in range(n):
    if (j%np.floor(0.05*n)==0)&(j>0):
        perc_complete = perc_complete+5.
        print('{:.1f}% complete'.format(perc_complete))
    i=sample[j]
    input_err=gsi_err1[i]
    adjust_err=gsi_err2[i]
    final_err=gsi_err3[i]
    ob_typ=ob_typu[i]
    ob_qm=gsi_qm_u[i]
    ob_p=ob_pre[i]
    ob_ps=geo_ps[i]
    ob_p_prof=geo_lev[i,:].squeeze()
    ob_tp=geo_tp[i]
    ob_isli=geo_isli[i]
    ob_speed=ob_spd[i]
    bk_speed=ufo_spd[i]
    ob_direc=ob_dir[i]
    bk_direc=ufo_dir[i]
    ob_u_omf=ob_u[i]-ufo_u[i]
    ob_v_omf=ob_v[i]-ufo_v[i]

    qc_test,qc_flag=simulate_GSI_QC_satwinds(input_err,adjust_err,ob_typ,ob_qm,ob_p,ob_ps,ob_p_prof,
                                     ob_tp,ob_isli,ob_speed,bk_speed,ob_direc,bk_direc,ob_u_omf,ob_v_omf,convinfo)
    if qc_test:
        sim_qc_u[j]=0.
    else:
        sim_qc_u[j]=1.
    sim_qc_flg.append(qc_flag)

perc_complete = perc_complete+5.
print('{:.1f}% complete'.format(perc_complete))
sim_qc_u=sim_qc_u.astype('int32')

np.corrcoef(gsi_qc_u[0:n],sim_qc_u)

