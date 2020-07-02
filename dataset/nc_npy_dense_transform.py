import xarray as xr
import numpy as np

def get_date_loc(date, loc): 
    dsh = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/HIM_proc/HIM8_{loc}_{date}.nc")
    #dsg = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/GPM_proc/GPM_{loc}_{date}.nc")
    dsg = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/GPM_proc/GPM_SYD_{date}.nc")
  
    print(dsh.time) 
    print(dsg.time) 

    not_nan = np.ones(dsg.time.shape)==1
    for band in [11, 16]: 
        x = dsh[f'B{band}'].resample(time="10Min").nearest(tolerance="1Min").values.astype(np.float32).reshape(-1,3,512,512)
        dsh.close()
        not_nan = ~np.isnan(x).any(axis=(1,2,3))* not_nan
        x = None
   
    #y = dsg['PrecCal'].sel(time=dsh.time).values.astype(np.float32)
    y = dsg['PrecCal'].values.astype(np.float32)
    y = y[not_nan,:,:]
    print(date, loc, y.shape)
    np.save(f"/g/data/fj4/scratch/Y_DENSE_{loc}_{date}.nc", y)
    dsg.close()

    for band in [11, 16]: 
        x = dsh[f'B{band}'].resample(time="10Min").nearest(tolerance="1Min").values.astype(np.float32).reshape(-1,3,512,512)
        dsh.close()
        x = x[not_nan,:,:,:]
        print(band, date, loc, x.shape)
        np.save(f"/g/data/fj4/scratch/X_DENSE_B{band}_{loc}_{date}.nc", x)


#for loc in ["SYD","NT","WA"]:
for loc in ["SE"]:
    for date in ["201811","201812","201901","201902","201903","201904"]:
        get_date_loc(date, loc)

