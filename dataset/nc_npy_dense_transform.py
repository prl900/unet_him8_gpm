import xarray as xr
import numpy as np

def get_date_loc(date, loc): 
    dsh = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/HIM_proc/HIM8_{loc}_{date}.nc")
    dsg = xr.open_dataset(f"/g/data/fj4/scratch/him_gpm_0/GPM_proc/GPM_{loc}_{date}.nc")
  
    print(dsh.time) 
    print(dsg.time) 

    not_nan = np.ones(dsh.time.shape)==1
    for band in range(8,17): 
        x = dsh[f'B{band}'].values.astype(np.float32)
        not_nan = ~np.isnan(x).any(axis=(1,2)) * not_nan
    
    y = dsg['PrecCal'].sel(time=dsh.time).values.astype(np.float32)
    y = y[not_nan,:,:]
    print(date, loc, y.shape)
    np.save(f"Y_{loc}_{date}.nc", y)
    dsg.close()

    for band in range(8,17): 
        x = dsh[f'B{band}'].values.astype(np.float32)
        dsh.close()
        x = x[not_nan,:,:]
        print(band, date, loc, x.shape)
        np.save(f"X_B{band}_{loc}_{date}.nc", x)


for loc in ["SYD","NT","WA"]:
    for date in ["201811","201812","201901","201902","201903","201904"]:
        get_date_loc(date, loc)

