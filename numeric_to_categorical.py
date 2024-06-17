import pandas as pd 
from sklearn.cluster import KMeans

#file path excel 
file_path = 'aksara_jawa_dataset.xlsx'

#load data file excel
data = pd.read_excel(file_path)

#untuk menyimpan centroids dari tiap kolom
centroids_cluster = {}

#inisialisasi dan melatih model kmeans untuk tiap kolom 
for column in data.columns:
    if column != 'cluster':
        if data[column].dtype in ['int64', 'float64']:  # Check if column is numeric
            cls = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan')
            column_data = data[[column]]
            cls.fit(column_data)
            labels = cls.labels_
            data[f'{column}_cluster'] = labels 
            #simpan centroids dari tiap kolom 
            centroids_cluster[column] = cls.cluster_centers_

#menampilkan centroid dari setiap kolom
print("Centroid dari tiap kolom: ")
for column, centroids in centroids_cluster.items():
    print(f"Kolom {column}:")
    for i, centroid in enumerate(centroids):
        print(f" Cluster {i}: {centroid[0]}")

data.to_csv('hasil.csv', index=False)

hasil = pd.read_csv('hasil.csv')
data = data.drop(columns=['bahasa_jawa', 'bahasa_indonesia', 'ipa', 'ips'])

data.to_csv('numeric_to_categorical.csv', index=False)

