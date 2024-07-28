import numpy as np
import pandas as pd

# load dataset
data = pd.read_excel('aksara_jawa_dataset.xlsx')

#kategori
categories = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']

# discritize semua data di excel
for column in data.select_dtypes(include=[np.number]).columns:
    min_value = data[column].min()
    max_value = data[column].max()
    
    bins = np.linspace(min_value, max_value, 6)  
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=categories, include_lowest=True)

#menghapus kolom bahasa_jawa, bahasa_indonesia, ipa, dan ips
data = data.drop(columns=['bahasa_jawa', 'bahasa_indonesia', 'ipa', 'ips'])

#simpan hasil ke csv
data.to_csv('discretize.csv', index=False)