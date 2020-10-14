import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
 
f, axarr = plt.subplots(4, 5, figsize=(8,5))

for i, event in enumerate([1284,66,241,925]):
    img = np.load(f'/data/GPM_HIM8/models/yhat_SYD_v0_b8_s1.npy')[event,:,:,0]
    print(img.shape)
    axarr[i,0].imshow(img, vmin=0, vmax=15, cmap=cm.Blues)
    if i == 0:
        axarr[i,0].set_title(f"Band 8") 
    axarr[i,0].axis('off')
    
    img = np.load(f'/data/GPM_HIM8/models/yhat_SYD_v0_b11_s1.npy')[event,:,:,0]
    axarr[i,1].imshow(img, vmin=0, vmax=15, cmap=cm.Blues)
    if i == 0:
        axarr[i,1].set_title(f"Band 11") 
    axarr[i,1].axis('off')
    
    img = np.load(f'/data/GPM_HIM8/models/yhat_SYD_v0_b16_s1.npy')[event,:,:,0]
    axarr[i,2].imshow(img, vmin=0, vmax=15, cmap=cm.Blues)
    if i == 0:
        axarr[i,2].set_title(f"Band 16") 
    axarr[i,2].axis('off')
    
    img = np.load(f'/data/GPM_HIM8/models/yhat_SYD_v0_b11_16_s1.npy')[event,:,:,0]
    axarr[i,3].imshow(img, vmin=0, vmax=15, cmap=cm.Blues)
    if i == 0:
        axarr[i,3].set_title(f"Bands 11 16") 
    axarr[i,3].axis('off')
    
    img = np.load(f"/data/GPM_HIM8/exp_paper/Y_SYD_201811.npy")[event,:,:]
    im = axarr[i,4].imshow(img, vmin=0, vmax=15, cmap=cm.Blues)
    if i == 0:
        axarr[i,4].set_title(f"IMERG") 
    axarr[i,4].axis('off')

f.subplots_adjust(right=1.8)
f.colorbar(im)
    
plt.show()
