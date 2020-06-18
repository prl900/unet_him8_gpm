module load gdal/2.2.2
module load cdo
module load nco

DATE=$(date -u -d "2019-03-01")
END=$(date -u -d "2019-04-01")
LOC="SE"
COUNTER=0

while [ "$DATE" != "$END" ]; do
	echo $DATE
	for BAND in {7..16}; do
		FILE="/g/data/rr5/satellite/obs/himawari8/FLDK/"$(date -u +%Y -d "$DATE")"/"$(date -u +%m -d "$DATE")"/"$(date -u +%d -d "$DATE")"/"$(date -u +%H%M -d "$DATE")"/"$(date -u +%Y%m%d%H%M -d "$DATE")"00-P1S-ABOM_OBS_B"$(printf %02d $BAND)"-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc"
		echo $FILE
		if [ -f $FILE ]; then
			if [ "$LOC" == "SE" ]; then
				gdalwarp -of netCDF -r bilinear -co WRITE_BOTTOMUP=NO -t_srs EPSG:3577 -te 900000 -4524000 1924000 -3500000 -tr 2000 -2000 $FILE B$BAND.nc
			fi
			if [ "$LOC" == "NT" ]; then
				gdalwarp -of netCDF -r bilinear -co WRITE_BOTTOMUP=NO -t_srs EPSG:3577 -te 65000 -2215000 1089000 -1191000 -tr 2000 -2000 $FILE B$BAND.nc
			fi
			if [ "$LOC" == "WA" ]; then
				gdalwarp -of netCDF -r bilinear -co WRITE_BOTTOMUP=NO -t_srs EPSG:3577 -te -1824000 -3680000 -800000 -2656000 -tr 2000 -2000 $FILE B$BAND.nc
			fi
			ncrename -v Band1,B$BAND B$BAND.nc
		fi
	done
		
	cdo merge B8.nc B9.nc B10.nc B11.nc B12.nc B13.nc B14.nc B15.nc B16.nc BS.nc
	cdo setdate,$(date -u +%Y-%m-%d -d "$DATE") BS.nc BD.nc
	cdo settime,$(date -u +%H:%M:%S -d "$DATE") BD.nc BE.nc
	cdo setreftime,1970-01-01 BE.nc BF.nc
	cdo settunits,seconds BF.nc BG.nc
	cdo setcalendar,standard BG.nc "HIM8_"$COUNTER"_AU.nc"
	rm B*.nc

	DATE=$(date -u -d "$DATE + 30 minute")
	COUNTER=$(( $COUNTER + 1 ))

        if [ $COUNTER -gt 100 ] || [ "$DATE" == "$END" ]; then
		cdo mergetime HIM8_*_AU.nc batch.nc
		if [ $? -ne 0 ]; then
    			continue
		fi
		rm HIM8_*_AU.nc

		if [ ! -f "HIM8_AU_BoM.nc" ]; then
			mv batch.nc "HIM8_AU_BoM.nc"
		else
			mv "HIM8_AU_BoM.nc" aux.nc
			cdo mergetime aux.nc batch.nc "HIM8_AU_BoM.nc"
			rm aux.nc
			rm batch.nc
		fi

		COUNTER=0
	fi

done
