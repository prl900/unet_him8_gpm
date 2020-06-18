#!/g/data/v10/public/modules/dea-env/20181015/bin/python

import argparse
import datetime
import netCDF4
import h5py
import numpy as np

def pack(data, date, dst):
    with netCDF4.Dataset(dst, 'w', format='NETCDF4_CLASSIC') as ds:
        setattr(ds, "date_created", datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        
        ds.createDimension("time", 1)
        ds.createDimension("latitude", data.shape[0])
        ds.createDimension("longitude", data.shape[1])

        var = ds.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1970-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num([date], units="seconds since 1970-01-01 00:00:00.0", calendar="standard")

        var = ds.createVariable("longitude", "f8", ("longitude",))
        var.units = "degrees_east"
        var.long_name = "longitude"
        var[:] = np.linspace(-179.95, 179.95, 3600)

        var = ds.createVariable("latitude", "f8", ("latitude",))
        var.units = "degrees_north"
        var.long_name = "latitude"
        var[:] = np.linspace(89.95, -89.95, 1800)

        var = ds.createVariable("precipitationCal", "f4", ("time", "latitude", "longitude"), fill_value=-9999.9)
        var.long_name = "Precipitation Calibrated"
        var.units = 'mm/h'
        var[:] = data[None,...]


def get_prec(f_path):
    with h5py.File(f_path, mode='r') as f:
        prec = f['Grid']['precipitationCal'][:].T[::-1, :]
        prec[prec == -9999.9] = np.nan
   
    return prec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""GPM HDF-EOS to netCDF4 converter""")
    parser.add_argument('-src', '--source', required=True, type=str, help="Full path to source.")
    parser.add_argument('-dst', '--destination', required=True, type=str, help="Full path to destination.")
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.source[74:82]+args.source[84:88], '%Y%m%d%H%M')
    print(date)
    prec = get_prec(args.source)
    pack(prec, date, args.destination)
