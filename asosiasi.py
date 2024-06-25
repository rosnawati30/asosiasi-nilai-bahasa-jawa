import pandas as pd
import matplotlib.pyplot as  plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('discretize.csv')

#menambahkan nama kolom ke setiap value
trx = data.apply(lambda x: x.name + '=' + x).values.tolist()

#encode transaksi 
trx_encod = TransactionEncoder()
trx_encod_ary = trx_encod.fit(trx).transform(trx)
data_trans = pd.DataFrame(trx_encod_ary, columns=trx_encod.columns_)

#menggunakan fp-growth dengna min_support = 0.2
frequent_itemset = fpgrowth(data_trans, min_support=0.2, use_colnames=True)
print(frequent_itemset)

# Menghitung rule asosiasi
rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.4)
print(rules)

#plotting
plt.figure(figsize=(10,6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Rule Asosiasi')

for i in range(len(rules)):
    plt.annotate(f"R{i+1}", (rules['support'][i], rules['confidence'][i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
