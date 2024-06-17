import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

data = pd.read_excel('nilai_bahasa_indonesia.xlsx')

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

clustering = KMeans(n_clusters=3, random_state=42)

clustering.fit(train_set)

print("Cluster centers (centroids): ")
print(clustering.cluster_centers_)

train_set['Cluster'] = clustering.predict(train_set)
test_set['Cluster'] = clustering.predict(test_set)

# Davies-Bouldin Index untuk data training
train_db_index = davies_bouldin_score(train_set.drop(columns=['Cluster']), clustering.labels_)

# Davies-Bouldin Index untuk data testing
test_db_index = davies_bouldin_score(test_set.drop(columns=['Cluster']), clustering.predict(test_set.drop(columns=['Cluster'])))

print("Davies-Bouldin Index (training set):", train_db_index)
print("Davies-Bouldin Index (testing set):", test_db_index)

