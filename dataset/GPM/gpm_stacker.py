module load gdal/2.2.2
module load cdo
module load nco

DATE=$(date -u -d "2018-12-01")
END=$(date -u -d "2019-01-01")
LOC="SE"
COUNTER=0

while [ "$DATE" != "$END" ]; do
	echo $COUNTER
	echo $DATE
	
	END_DATE=$(date -u -d "$DATE + 29 minutes + 59 seconds")
        MINS=$((10#$(date -u +%H -d "$DATE")*60+10#$(date -u +%M -d "$DATE")))
	FORMATTED_MINS=$(echo 00000$MINS | tail -c 5)
	FILE="/g/data/fj4/SatellitePrecip/GPM/global/late/"$(date -u +%Y%m -d "$DATE")"/3B-HHR-L.MS.MRG.3IMERG."$(date -u +%Y%m%d -d "$DATE")"-S"$(date -u +%H%M%S -d "$DATE")"-E"$(date -u +%H%M%S -d "$END_DATE")"."$FORMATTED_MINS".V06B.RT-H5"
	echo $FILE

	if [ ! -f $FILE ]; then
		DATE=$(date -u -d "$DATE + 30 minutes")
		continue
	fi

	./gpm_transformer.py -src $FILE -dst aux.nc

	if [ "$LOC" = "SE" ]; then
	gdalwarp -of netCDF -co WRITE_BOTTOMUP=NO -r bilinear -s_srs EPSG:4326 -t_srs EPSG:3577 -te 900000 -4524000 1924000 -3500000 -tr 8000 -8000 aux.nc BB.nc
	fi
	if [ "$LOC" = "NT" ]; then
	gdalwarp -of netCDF -r bilinear -co WRITE_BOTTOMUP=NO -s_srs EPSG:4326 -t_srs EPSG:3577 -te 65000 -2215000 1089000 -1191000 -tr 8000 -8000 aux.nc BB.nc
	fi
	if [ "$LOC" = "WA" ]; then
	gdalwarp -of netCDF -r bilinear -co WRITE_BOTTOMUP=NO -s_srs EPSG:4326 -t_srs EPSG:3577 -te -1824000 -3680000 -800000 -2656000 -tr 8000 -8000 aux.nc BB.nc
	fi

	rm aux.nc
        ncrename -v Band1,PrecCal BB.nc

        cdo setdate,$(date -u +%Y-%m-%d -d "$DATE") BB.nc BD.nc
        cdo settime,$(date -u +%H:%M:%S -d "$DATE") BD.nc BE.nc
        cdo setreftime,1970-01-01 BE.nc BF.nc
        cdo settunits,seconds BF.nc BG.nc
        cdo setcalendar,standard BG.nc C$COUNTER.nc 
	
        rm B*.nc

	DATE=$(date -u -d "$DATE + 30 minutes")
	COUNTER=$(( $COUNTER + 1 ))

        if [ $COUNTER -gt 100 ] || [ "$DATE" == "$END" ]; then
		cdo mergetime C*.nc batch.nc
		rm C*.nc
	    	
		if [ ! -f "GPM_BoM_"$(date -u +%Y%m -d "$DATE - 30 minutes")".nc" ]; then
			mv batch.nc "GPM_BoM_"$(date -u +%Y%m -d "$DATE - 30 minutes")".nc"
		else
			mv "GPM_BoM_"$(date -u +%Y%m -d "$DATE - 30 minutes")".nc" aux.nc
			cdo mergetime aux.nc batch.nc "GPM_BoM_"$(date -u +%Y%m -d "$DATE - 30 minutes")".nc"
			rm aux.nc
			rm batch.nc
		fi

		cdo infon "GPM_BoM_"$(date -u +%Y%m -d "$DATE - 30 minutes")".nc"
		
		COUNTER=0
	fi


done
