from tensorflow.keras.models import load_model
import numpy as np

def get_band_data3(loc, dates, b, mean=None, std=None):
    y = np.concatenate([np.load(f"/data/GPM_HIM8/exp_paper/Y_{loc}_{d}.npy") for d in dates], axis=0)
    y = np.clip(y,0,30)

    x11 = np.concatenate([np.load(f"/data/GPM_HIM8/exp_paper/X_B11_{loc}_{d}.npy") for d in dates], axis=0)
    x16 = np.concatenate([np.load(f"/data/GPM_HIM8/exp_paper/X_B16_{loc}_{d}.npy") for d in dates], axis=0)
    xi = np.concatenate([np.load(f"/data/GPM_HIM8/exp_paper/X_B{b}_{loc}_{d}.npy") for d in dates], axis=0)

    if mean is None:
        mean = [x11.mean(),x16.mean(),xi.mean()]
        std = [x11.std(),x16.std(),xi.std()]

    x11 = (x11-mean[0])/std[0]
    x16 = (x16-mean[1])/std[1]
    xi = (xi-mean[2])/std[2]

    x = np.stack((x11,x16,xi), axis=3)
    x11 = None
    x16 = None
    xi = None

    return x, y[:,:,:,None], mean, std

dates = ["201811","201812","201901","201902"]

for loc in ["SYD", "NT", "WA"]:
    for val in range(4):
        for b in range(8,17):
            _, _, mean, std = get_band_data3(loc, [x for i, x in enumerate(dates) if i!=val], b)
            x_test, y_test, _, _ = get_band_data3(loc, [x for i, x in enumerate(dates) if i==val], b, mean, std)
            
            for i in range(5):
                print(loc, val, b, i)
                mod = load_model(f'/data/GPM_HIM8/models/model_3months_200epochs_8chann_v{val}_{loc}_s{i+1}_b11_16_{b}.h5', 
                                 custom_objects={'accuracy05':accuracy05, 'precision05':precision05, 'recall05':recall05,
                                                 'accuracy1':accuracy1, 'precision1':precision1, 'recall1':recall1,
                                                 'accuracy5':accuracy5, 'precision5':precision5, 'recall5':recall5,
                                                 'accuracy10':accuracy10, 'precision10':precision10, 'recall10':recall10})

                yhat = mod.predict(x_test)
                np.save(f'/data/GPM_HIM8/models/yhat_{loc}_v{val}_b11_16_{b}_s{i+1}', yhat)
