import numpy as np
import pandas as pd

def stat_generator(bands):
    
    dates = ["201811","201812","201901","201902"]
    data = []
    
    for loc in ["SYD","WA","NT"]:
        print(loc)
        for b in range(8,17):
            print(b)
            scores = []
            for val_i, d in enumerate(dates):
                y_test = np.load(f"/data/GPM_HIM8/exp_paper/Y_{loc}_{d}.npy")
                y_test = np.clip(y_test,0,30)

                for i in range(5):
                    if bands=='1B':
                        yhat = np.load(f'/data/GPM_HIM8/models/yhat_{loc}_v{val_i}_b{b}_s{i+1}.npy')
                    elif bands=='2B':
                        yhat = np.load(f'/data/GPM_HIM8/models/yhat_{loc}_v{val_i}_b11_{b}_s{i+1}.npy')
                    elif bands=='3B':
                        yhat = np.load(f'/data/GPM_HIM8/models/yhat_{loc}_v{val_i}_b11_16_{b}_s{i+1}.npy')

                    pred = yhat[:,:,:,0]
                    obs = y_test[:,:,:]
                    mse = np.mean(np.square(obs-pred))

                    for t in [0.1,0.2,0.5,1.0,2.0,5.0]:
                        pred = yhat[:,:,:,0]>t
                        obs = y_test[:,:,:]>t
                        prec = (obs*pred).sum()/pred.sum()
                        rec = (obs*pred).sum()/obs.sum()
                        f1 = (2*rec*prec)/(rec+prec)

                        data.append({'Loc': loc,
                                     'Band': f'B{b}',
                                     'MSE': mse,
                                     'Rec': rec,
                                     'Prec': prec,
                                     'F1': f1,
                                     'threshold [mm/h]': t})

        
    return pd.DataFrame(data)


for bands in ["1B","2B","3B"]:
    data = stat_generator(bands=bands)
    data.to_csv(f"results_{bands}.csv")
