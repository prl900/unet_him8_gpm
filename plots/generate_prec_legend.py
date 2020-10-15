from pylab import figure, cm
from matplotlib.colors import LogNorm

C = np.load(f'/data/GPM_HIM8/models/yhat_SYD_v0_b8_s3.npy')[1284,:,:,0]

f = figure(figsize=(6.2,5.6))
ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
#im = ax.matshow(C, cmap=newcmp, norm=LogNorm(vmin=0.01, vmax=15))
im = ax.imshow(C, cmap=newcmp, norm=LogNorm(vmin=0.01, vmax=15))
t = [0.01, 0.1, 0.2, 0.5, 1.0, 2, 5, 10, 15]
f.colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
f.show()
