import pandas as pd
import numpy as np
import pickle 
from matplotlib import pyplot as plt

def compute_crr(loc, date):
    CRRCoefficients = [9.0e+08,
                        -8.28687176e-02,
                       1.90003955e-01,
                       -4.6e+01,
                       3.99996036e+00,
                       -2.3e+02,
                       3.99997842e+00,
                       2.99966504e+00]

    ir = np.load(f"/data/GPM_HIM8/exp_paper/X_B14_{loc}_{date}.npy")[:,::4,::4]
    wv = np.load(f"/data/GPM_HIM8/exp_paper/X_B8_{loc}_{date}.npy")[:,::4,::4]
    diff = ir-wv
    
    H_IR = CRRCoefficients[0] * np.exp(ir * CRRCoefficients[1])

    C_IR = CRRCoefficients[2] * ir + CRRCoefficients[3]

    W_IR = CRRCoefficients[4] * \
           np.exp( -0.5 * ((ir + CRRCoefficients[5]) / CRRCoefficients[6]) ** 2.0) + \
           CRRCoefficients[7] 

    return H_IR * np.exp(-0.5 * ((((diff) - C_IR) / W_IR) ** 2.0))

def gpm_prec(loc, date):
    return np.clip(np.load(f"/data/GPM_HIM8/exp_paper/Y_{loc}_{date}.npy"),0,40)

def cnn_prec(loc, date, b1, b2, seed):
    i = ["201811","201812","201901","201902"].index(date)
    return np.clip(np.load(f"/data/GPM_HIM8/models/yhat_{loc}_v{i}_b{b1}_{b2}_s{seed}.npy"),0,40)

def rf_prec(loc, date):
    rf = pickle.load(open(f'baseline_models/rf_{loc}_{date}.pkl', 'rb'))

    b11 = np.load(f"/data/GPM_HIM8/exp_paper/X_B11_{loc}_{date}.npy")[:,::4,::4]
    b16 = np.load(f"/data/GPM_HIM8/exp_paper/X_B16_{loc}_{date}.npy")[:,::4,::4]
    
    x = np.stack((b11.flatten(), b16.flatten()),axis=1)

    return rf.predict(x).reshape((-1,128,128))


loc = "WA"
date = "201901"
      
gpm = gpm_prec(loc, date)
crr = compute_crr(loc, date)
rf = rf_prec(loc, date)
cnn = cnn_prec(loc, date, 11, 16, 1)[:,:,:,0]


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap

red = np.array([255, 252, 250, 247, 244, 242, 239, 236, 234, 231, 229, 226, 223, 221, 218, 215, 213, 210,
                     207, 205, 202, 199, 197, 194, 191, 189, 186, 183, 181, 178, 176, 173, 170, 168, 165, 162,
                     157, 155, 152, 150, 148, 146, 143, 141, 139, 136, 134, 132, 129, 127, 125, 123, 120, 118,
                     116, 113, 111, 109, 106, 104, 102, 100, 97,  95,  93,  90,  88,  86,  83,  81,  79,  77,
                     72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,
                     72,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,
                     73,  78,  83,  87,  92,  97,  102, 106, 111, 116, 121, 126, 130, 135, 140, 145, 150, 154,
                     159, 164, 169, 173, 178, 183, 188, 193, 197, 202, 207, 212, 217, 221, 226, 231, 236, 240,
                     245, 250, 250, 250, 250, 249, 249, 249, 249, 249, 249, 249, 249, 248, 248, 248, 248, 248,
                     248, 248, 247, 247, 247, 247, 247, 247, 247, 246, 246, 246, 246, 246, 246, 246, 246, 245,
                     245, 245, 244, 243, 242, 241, 240, 239, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230,
                     229, 228, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 217, 216, 215, 214,
                     213, 211, 209, 207, 206, 204, 202, 200, 199, 197, 195, 193, 192, 190, 188, 186, 185, 183,
                     181, 179, 178, 176, 174, 172, 171, 169, 167, 165, 164, 162, 160, 158, 157, 155, 153, 151, 150, 146], dtype = np.float)

red = red / 255

green = np.array([255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238,
                     237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
                     218, 216, 214, 212, 210, 208, 206, 204, 202, 200, 197, 195, 193, 191, 189, 187, 185, 183,
                     181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 160, 158, 156, 154, 152, 150, 148, 146,
                     142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160,
                     161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179,
                     181, 182, 184, 185, 187, 188, 189, 191, 192, 193, 195, 196, 198, 199, 200, 202, 203, 204,
                     206, 207, 209, 210, 211, 213, 214, 215, 217, 218, 220, 221, 222, 224, 225, 226, 228, 229,
                     231, 232, 229, 225, 222, 218, 215, 212, 208, 205, 201, 198, 195, 191, 188, 184, 181, 178,
                     174, 171, 167, 164, 160, 157, 154, 150, 147, 143, 140, 137, 133, 130, 126, 123, 120, 116,
                     113, 106, 104, 102, 100,  98,  96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,
                     72,  70,  67,  65,  63,  61,  59,  57,  55,  53,  51,  49,  47,  45,  43,  41,  39,  37,
                     35,  31,  31,  30,  30,  30,  30,  29,  29,  29,  29,  28,  28,  28,  27,  27,  27,  27,
                     26,  26,  26,  26,  25,  25,  25,  25,  24,  24,  24,  23,  23,  23,  23,  22,  22,  22, 22,  21], dtype = np.float)

green = green / 255

blue = np.array([255, 255, 255, 254, 254, 254, 254, 253, 253, 253, 253, 253, 252, 252, 252, 252, 252, 251,
                     251, 251, 251, 250, 250, 250, 250, 250, 249, 249, 249, 249, 249, 248, 248, 248, 248, 247,
                     247, 246, 245, 243, 242, 241, 240, 238, 237, 236, 235, 234, 232, 231, 230, 229, 228, 226,
                     225, 224, 223, 221, 220, 219, 218, 217, 215, 214, 213, 212, 211, 209, 208, 207, 206, 204,
                     202, 198, 195, 191, 188, 184, 181, 177, 173, 170, 166, 163, 159, 156, 152, 148, 145, 141,
                     138, 134, 131, 127, 124, 120, 116, 113, 109, 106, 102, 99,  95,  91,  88,  84,  81,  77,
                     70,  71,  71,  72,  72,  73,  74,  74,  75,  75,  76,  77,  77,  78,  78,  79,  80,  80,
                     81,  81,  82,  82,  83,  84,  84,  85,  85,  86,  87,  87,  88,  88,  89,  90,  90,  91,
                     91,  92,  91,  89,  88,  86,  85,  84,  82,  81,  80,  78,  77,  75,  74,  73,  71,  70,
                     69,  67,  66,  64,  63,  62,  60,  59,  58,  56,  55,  53,  52,  51,  49,  48,  47,  45,
                     44,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,
                     41,  41,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,
                     40,  40,  40,  39,  39,  38,  38,  38,  37,  37,  36,  36,  36,  35,  35,  34,  34,  34,
                     33,  33,  32,  32,  31,  31,  31,  30,  30,  29,  29,  29,  28,  28,  27,  27,  27,  26, 26,  25], dtype = np.float)

blue = blue / 255

N = 254
vals = np.ones((N, 4))
vals[:, 0] = red
vals[:, 1] = green
vals[:, 2] = blue

prec_cm = ListedColormap(vals)


import cartopy.crs as ccrs

aaea = ccrs.AlbersEqualArea(central_latitude=0,
                            false_easting=0,
                            false_northing=0,
                            central_longitude=132,
                            standard_parallels=(-18, -36) )


img_extents = {"SYD": [900000, 1924000, -4524000, -3500000],
               "NT": [65000, 1089000, -2215000, -1191000],
               "WA": [-1824000, -800000, -3680000, -2656000]}


def generate_plot(loc, imgs, date, t):
    fig = plt.figure(figsize=(24, 10))

    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i+1, projection= aaea)

        ax.imshow(imgs[i], origin='upper', extent=img_extents[loc], cmap=prec_cm, norm=LogNorm(vmin=0.01, vmax=15))

        ax.set_extent(img_extents[loc], crs=aaea)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.8, linestyle='--')
        gl.xlocator = mticker.FixedLocator([110,115,120,125,130,135,140,145,150,155])
        gl.ylocator = mticker.FixedLocator([-10,-15,-20,-25,-30, -35, -40])
        gl.xlabel_style = {'size': 15}
        gl.ylabel_style = {'size': 15}

    plt.savefig(f"baseline_{loc}_{date}_{t:04d}.png")
    

for t in range(gpm.shape[0]):
    generate_plot(loc, [gpm[t], cnn[t], crr[t], rf[t]], date, t)
