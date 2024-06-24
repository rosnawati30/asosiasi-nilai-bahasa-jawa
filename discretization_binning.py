import numpy as np
import pandas as pd

# load dataset
data = pd.read_excel('aksara_jawa_dataset.xlsx')

#menentukan lima kategori 
# 0-20 (Sangat Rendah), 21-40 (Rendah), 41-60 (Sedang), 61-80 (Tinggi), 81-100 (Sangat Tinggi)
thresholds = [0, 20, 40, 60, 80, 100]
categories = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']

# discritize semua data di excel
for column in data.select_dtypes(include=[np.number]).columns:
    min_value = data[column].min()
    max_value = data[column].max()
    # Create thresholds dynamically based on the column's min and max values
    bins = np.linspace(min_value, max_value, 6)  # 5 bins means 6 edges
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=categories, include_lowest=True)

#simpan hasil ke csv
data.to_csv('discretize.csv', index=False)
