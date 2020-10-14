import matplotlib.pyplot as plt

from shapely import geometry

import cartopy.crs as ccrs

aaea = ccrs.AlbersEqualArea(central_latitude=0,
                            false_easting=0,
                            false_northing=0,
                            central_longitude=132,
                            standard_parallels=(-18, -36) )

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1, projection= aaea)

ax.stock_img()

ax.set_extent([110, 155, -44, -8])
ax.coastlines()
ax.gridlines(draw_labels=True)

geom = geometry.box(minx=-1824000, maxx=-800000, miny=-3680000, maxy=-2656000)
ax.add_geometries([geom], crs=aaea, color='red', alpha=0.6)

geom = geometry.box(minx=65000, maxx=1089000, miny=-2215000, maxy=-1191000)
ax.add_geometries([geom], crs=aaea, color='red', alpha=0.6)

geom = geometry.box(minx=900000, maxx=1924000, miny=-4524000, maxy=-3500000)
ax.add_geometries([geom], crs=aaea, color='red', alpha=0.6)

plt.title('World in Albers Equal Area (centered on Australia)', {'fontsize':30}, pad=40)

plt.show()
