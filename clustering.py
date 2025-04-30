import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from helpers import *

payload_var='age'
cross_tab_var='state'
dname = 'df_fuzzy_names_5k_lsh200-50-100'
dataset2 = pd.read_pickle("dataset/"+ dname + ".pkl")

#print(dataset2.head())

##Norm Data
sign_norms = dataset2["Fuzzy Signature_Norm-200"]
data_norms = np.array([x for x in sign_norms.to_numpy()])

scaler = StandardScaler()
data_norms_scaled = scaler.fit_transform(data_norms)



# Example usage:
np.random.seed(42)

# Set the number of clusters
no_of_clusters = 50
name = dname + "_c" + str(no_of_clusters)
print(name)

# Run K-means clustering
cluster_centroids, cids = kmeans_dot_product(data_norms_scaled, no_of_clusters)
dataset2['Cluster_Id'] = cids

count = {i: 0 for i in range(no_of_clusters)}
for i in cids:
    count[i] += 1    
    
_min = count[min(count, key=count.get)]
_max = count[max(count, key=count.get)]
print(count, _min, _max)

completed_items = []
for i in range(_max):
    count = {k: v - 1 if v > 0 else 0 for k, v in count.items()}
    completed_items.append(100000 - sum(count.values()))

print(completed_items)

plt.hist(cids, bins=np.arange(cids.min(), cids.max(), 1) )
plt.savefig("out/" + name + '.eps', format='eps')

print("Clustering Done")


# Padding
max_cluster_size = dataset2.groupby('Cluster_Id').size().max()
print("max_cluster_size", max_cluster_size)

# Initialize a DataFrame with the necessary columns
columns = ['Cluster_Id'] + [f'Item_{i}' for i in range(max_cluster_size)]
cluster_dataset2 = pd.DataFrame(columns=columns)
cluster_dataset2_IDs = pd.DataFrame(columns=columns)

dummy_element = np.ones(50)

for cluster_num in range(no_of_clusters):  # Cluster number
    cluster_items = dataset2[dataset2['Cluster_Id'] == cluster_num]['Fuzzy Signature_Norm-50'].tolist()
    cluster_items_IDs = dataset2[dataset2['Cluster_Id'] == cluster_num]['ID'].tolist()    
    #print(cluster_num, len(cluster_items))
    
    while len(cluster_items) < max_cluster_size:
    #_len = max_cluster_size - len(cluster_items)
        cluster_items.append(dummy_element)
        cluster_items_IDs.append("NULL")
    
    # Assign the cluster items to separate columns
    data = [cluster_num] + cluster_items
    cluster_dataset2.loc[cluster_num] = data
    
    IDs = [cluster_num] + cluster_items_IDs
    cluster_dataset2_IDs.loc[cluster_num] = IDs

cluster_dataset2.to_pickle("out/" + name + ".pkl")
cluster_dataset2_IDs.to_pickle("out/" + name + "_IDs.pkl")
np.save("out/" + name + "_centroids.npy", cluster_centroids)

print("Padding Done")



# Create a dictionary that maps IDs to their corresponding payload values
id_to_payload = dict(zip(dataset2['ID'], dataset2[payload_var]))

# Function to safely get payload values, treating "NULL" as missing
def safe_get_payload(x):
    if x == "NULL":  # If ID is "NULL", return 0
        return 0
    return id_to_payload.get(x, 0)  # Default to 0 if ID is not found

# Initialize an empty array to store the payload values corresponding to the IDs
payload_array = np.vectorize(safe_get_payload)(cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy())

def to_float_or_zero(value):
    try:
        return float(value)
    except ValueError:
        return 0.0

# Vectorize the conversion function
vectorized_to_float_or_zero = np.vectorize(to_float_or_zero)

payload_dataset2_array=vectorized_to_float_or_zero(payload_array)
payload_dataset2=pd.DataFrame(payload_dataset2_array)
payload_dataset2.to_pickle("out/" + name + "_payload.pkl")

# Create a dictionary that maps IDs to their corresponding payload values
id_to_cross_tab_var = dict(zip(dataset2['ID'], dataset2[cross_tab_var].apply(lambda x:str(x))))

# Initialize an empty array to store the payload values corresponding to the IDs
cross_tab_array = np.vectorize(id_to_cross_tab_var.get)(cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy())


# Vectorize the conversion function
vectorized_to_float_or_zero = np.vectorize(to_float_or_zero)

cross_tab_dataset2_array=vectorized_to_float_or_zero(cross_tab_array)
cross_tab_dataset2=pd.DataFrame(cross_tab_dataset2_array)
cross_tab_dataset2.to_pickle("out/" + name + "_cross_tab.pkl")



# Create a dictionary to hold the new DataFrames
df_dic = {}
# Loop through each unique value and create a DataFrame for each cross_tab_var
# Get the unique values in the column
freq_list=list(cross_tab_dataset2.values.flatten())
unique_values = [x for x in freq_list if x!='None']
for value in unique_values:
    df=cross_tab_dataset2.applymap(lambda x: 1 if x==value else 0)
    df_dic[value]=df


with open("out/" + name + "_df_dic.pkl", 'wb') as output:
    pickle.dump(df_dic, output)