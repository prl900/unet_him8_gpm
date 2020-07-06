import xarray as xr
import numpy as np

def get_date_loc(date, loc): 
    dsh = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/HIM_proc/dense/HIM8_{loc}_{date}.nc")
    dsg = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/GPM_proc/GPM_{loc}_{date}.nc")

    print(dsh.time)
    print(dsg.time)
    print(dsg.time.shape)

    #Some files have duplicates in the time index 
    _, index = np.unique(dsh.time, return_index=True) 
    dsh = dsh.isel(time=index) 
    print(dsg.time.shape)

    #Some himawari files do not start at the begining of the month
    dsg = dsg.isel(time=slice(np.where(dsg.time==dsh.time[0])[0][0],None))
    print(dsg.time.shape)

    not_nan = np.ones(dsg.time.shape)==1
    for band in [11, 16]: 
        x = dsh[f'B{band}'].resample(time="10Min", ).nearest(tolerance="1Min").values.astype(np.float32).reshape(-1,3,512,512)
        dsh.close()
        not_nan = ~np.isnan(x).any(axis=(1,2,3))* not_nan
        x = None
   
    y = dsg['PrecCal'].values.astype(np.float32)
    y = y[not_nan,:,:]
    print(date, loc, y.shape)
    np.save(f"/g/data/fj4/scratch/Y_DENSE_{loc}_{date}", y)
    dsg.close()

    for band in [11, 16]: 
        x = dsh[f'B{band}'].resample(time="10Min").nearest(tolerance="1Min").values.astype(np.float32).reshape(-1,3,512,512)
        dsh.close()
        x = x[not_nan,:,:,:]
        print(band, date, loc, x.shape)
        np.save(f"/g/data/fj4/scratch/X_DENSE_B{band}_{loc}_{date}", x)


for loc in ["SE","NT","WA"]:
    for date in ["201811","201812","201901","201902"]:#,"201903","201904"]:
        get_date_loc(date, loc)

