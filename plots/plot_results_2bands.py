import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()

loc_map = {"SYD":"SE","NT":"NT","WA":"WA"}
scr_map = {"MSE":"MSE","F1":"F1","Rec":"Recall","Prec":"Precision"}
band_map = {'B8': '6.06–6.43 μm',
            'B9': '6.89–7.01 μm',
            'B10': '7.26–7.43 μm',
            'B11': '8.44–8.76 μm',
            'B12': '9.54–9.72 μm',
            'B13': '10.3–10.6 μm',
            'B14': '11.1–11.3 μm',
            'B15': '12.2–12.5 μm',
            'B16': '13.2–13.4 μm'}

bands = "2B"
thrs = 0.2

#------------------ Combined (Normalised) ----------------#
data_ref = pd.read_csv(f"results_1B.csv")
data = pd.read_csv(f"results_{bands}.csv")

data['Spectral Range'] = data['Band']
for b in range(8,17):
    data['Spectral Range'] = np.where(data['Spectral Range'] == f'B{b}', band_mapping['B11']+'\n'+band_mapping[f'B{b}'], data['Spectral Range'])

data = data[data.Band != "B11"]

se_med = data_ref[data_ref['threshold [mm/h]'] == thrs][data_ref['Loc'] == 'SYD'][data_ref['Band'] == 'B15'].F1.median()
data['F1'] = np.where((data['threshold [mm/h]'] == thrs) & (data['Loc'] == 'SYD'), data['F1']/se_med, data['F1'])
wa_med = data_ref[data_ref['threshold [mm/h]'] == thrs][data_ref['Loc'] == 'WA'][data_ref['Band'] == 'B15'].F1.median()
data['F1'] = np.where((data['threshold [mm/h]'] == thrs) & (data['Loc'] == 'WA'), data['F1']/wa_med, data['F1'])
nt_med = data_ref[data_ref['threshold [mm/h]'] == thrs][data_ref['Loc'] == 'NT'][data_ref['Band'] == 'B15'].F1.median()
data['F1'] = np.where((data['threshold [mm/h]'] == thrs) & (data['Loc'] == 'NT'), data['F1']/nt_med, data['F1'])

plt.clf()
plt.figure(figsize=(16,8))
ax = sns.boxplot(data=data[data['threshold [mm/h]'] == thrs], x='Spectral Range', y=scr)
ax.set_title(f"Validation normalised {scr_map[scr]} score at {thrs} [mm\h]")
plt.plot()


#------------------ By location ----------------#
data = pd.read_csv(f"results_{bands}.csv")

data['Spectral Range'] = data['Band']
for b in range(8,17):
    data['Spectral Range'] = np.where(data['Spectral Range'] == f'B{b}', band_mapping['B11']+'\n'+band_mapping[f'B{b}'], data['Spectral Range'])

data = data[data.Band != "B11"]

plt.clf()
plt.figure(figsize=(16,8))
ax = sns.boxplot(data=data[data['threshold [mm/h]'] == thrs], x='Spectral Range', y=scr, hue='Loc')
ax.set_title(f"Validation normalised {scr_map[scr]} score at {thrs} [mm\h]")
plt.plot()
