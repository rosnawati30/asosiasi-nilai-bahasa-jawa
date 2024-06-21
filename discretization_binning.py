import numpy as np 
import pandas as pd 

data = pd.read_excel('aksara_jawa_dataset.xlsx')

bins = np.linspace(min(data['bahasa_jawa']), max(data['bahasa_jawa']), 4)
print(bins)

kategori = ['Rendah', 'Sedang', 'Tinggi']

data['nilai_binned'] = pd.cut(data['bahasa_jawa'], bins, labels=kategori, include_lowest=True)

print(data['nilai_binned'])