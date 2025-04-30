
import socket
import numpy as np
import pandas as pd
import tenseal as ts
import time
import random
from tqdm import tqdm
from helpers import *


MIN_QUERY_SIZE=5

# Step 0: Load all required datasets
dataset_paths = {
    "centroids": "out/df_fuzzy_names_10k_lsh200-50-100_c50_centroids.npy",  # Centroids file
    "dataset2": "out/df_fuzzy_names_10k_lsh200-50-100_c50.pkl",  # Main dataset
    "payload": "out/df_fuzzy_names_10k_lsh200-50-100_c50_payload.pkl",  # Payload data
    "dataset3": "out/df_fuzzy_names_10k_lsh200-50-100_c50_cross_tab.pkl",  # Cross-tab dataset
    "df_dic": "out/df_fuzzy_names_10k_lsh200-50-100_c50_df_dic.pkl",  # Dictionary for DataFrame lookups
    "IDs": "out/df_fuzzy_names_10k_lsh200-50-100_c50_IDs.pkl"  # IDs dataset
}

datasets = {key: pd.read_pickle(path) if 'pkl' in path else np.load(path) for key, path in dataset_paths.items()}

# Extract loaded data into variables for easier access
cluster_centroids = datasets["centroids"]
cluster_dataset2 = datasets["dataset2"]
payload_dataset2 = datasets["payload"]
cluster_dataset3 = cluster_dataset2.drop('Cluster_Id', axis=1)  # Drop unnecessary column
cluster_dataset2_IDs = datasets["IDs"].drop('Cluster_Id', axis=1).to_numpy()  # Convert IDs to numpy array
cross_tab_dataset2 = datasets["dataset3"]
df_dic = datasets["df_dic"]

# Variables: Calculating the sizes for clusters
max_cluster_size = len(cluster_dataset2.columns) - 1  # Exclude 'Cluster_Id' column
no_of_clusters = len(cluster_centroids)  # Number of centroids (clusters)

# Network Setup
server_socket = socket.socket()
server_socket.bind(('localhost', 1234))  # Binding the server socket to a port
server_socket.listen(1)  # Listen for incoming connections
print("Server listening...")

# Accept incoming client connection
conn, addr = server_socket.accept()
print("Client connected:", addr)

print("Record linkage protocol started...")
send_data(conn, no_of_clusters)

# Step 1: Receive context from client
context_bytes = recv_data(conn)
context = ts.context_from(context_bytes)

enc_querier_bytes = recv_data(conn)
enc_querier_scaled = ts.ckks_tensor_from(context, enc_querier_bytes)  


# Step 2: Encrypt server-side data and compute
start_time = time.time()
enc_querier_scaled.mul_(cluster_centroids)
enc_querier_scaled.sum_(axis=2)

# Step 3: Send result back
send_data(conn, enc_querier_scaled.serialize())

#Step 2: perform computation on the most matching centroid 
enc_sign_of_centroids=recv_data(conn)
enc_sign_of_centroids=ts.ckks_tensor_from(context,enc_sign_of_centroids)

send_data(conn,max_cluster_size)


enc_querier_bytes = recv_data(conn)
enc_querier = ts.ckks_tensor_from(context, enc_querier_bytes)  

for i in tqdm(range(max_cluster_size)):
    col = np.array([x for x in cluster_dataset3["Item_"+str(i)].to_numpy()]).transpose()

    inner_time = time.time()
    
    enc_sign_of_centroids_tmp = enc_sign_of_centroids + 0
    
    enc_sign_of_centroids_tmp.mul_(col).sum_(axis=2)    
    enc_sign_of_centroids_tmp.mul_(enc_querier).sum_(axis=1)
    
    
    send_data(conn, enc_sign_of_centroids_tmp.serialize())
    

record_linkage_time=time.time() - start_time
print("Record linkage time: ",record_linkage_time)  

send_data(conn,cluster_dataset2_IDs)

print("Subquery size protocol started...")
enc_payload_querier=[]
for i in range(max_cluster_size):
    enc_payload_querier_bytes = recv_data(conn)
    payload_querier = ts.ckks_tensor_from(context, enc_payload_querier_bytes)  
    enc_payload_querier.append(payload_querier)
        

#print("<-")

#min query size protocol performed by B 

start_time=time.time()

sub_query_protocol=[]
sub_query_sizes={}
for key in df_dic.keys():
    sub_query_size=0
    for i in range(max_cluster_size):
        v2=ts.plain_tensor(df_dic[key].to_numpy()[:,i],[no_of_clusters,1])
        sub_query_size += enc_payload_querier[i].dot(v2)
    delta_query1=random.uniform(0, 100)*(sub_query_size-MIN_QUERY_SIZE)
    delta_query2=random.uniform(0, 100)*(MIN_QUERY_SIZE-sub_query_size)
    
    
    sub_query_protocol.append(delta_query1)
    sub_query_protocol.append(delta_query2)
    sub_query_sizes[key]=sub_query_size
    

n=len(sub_query_protocol)
permutation=random.sample(range(n), n)
inverse_perm=inverse_permutation(permutation)


sub_query_sizes_perm=[sub_query_protocol[i] for i in permutation]

subquery_time=time.time()-start_time
print("Subquery protocol time: ",subquery_time)

send_data(conn,n)

for x in sub_query_sizes_perm:
    send_data(conn, x.serialize())

sub_query_sizes_true=recv_data(conn)
sub_query_sizes_true=[sub_query_sizes_true[i] for i in inverse_perm]
#print("Sub query sizes: ",sub_query_sizes_true)
#print('Checks if alternate values are positive')

if check_alternating_signs(sub_query_sizes_true): 
    print("Subquery protocol passed")
    print('Main computation started...')
    start_time=time.time()
    cross_tab_dfs={}
    send_data(conn,len(df_dic.keys()))

    for key in df_dic.keys():
        cross_tab_dfs[key]=payload_dataset2*df_dic[key]

    cross_table_num=[]
    cross_table_dec=[]
    rand_list=[]
    values = np.linspace(0, 1, 100)

    for key in df_dic.keys():
        enc_tmp=0
        dataset=cross_tab_dfs[key]
        for i in range(max_cluster_size):
                v2=ts.plain_tensor(dataset.to_numpy()[:,i],[no_of_clusters,1])
                enc_res = enc_payload_querier[i].dot(v2)
                enc_tmp += enc_res
        

        # Select one random value from the 100 generated values
        r = np.random.choice(values)
        rand_list.append(r)
        cross_table_num.append(r*enc_tmp)
        cross_table_dec.append(r*sub_query_sizes[key])
        num=r*enc_tmp
        den=r*sub_query_sizes[key]
        
        send_data(conn,key)
        send_data(conn, num.serialize())
        send_data(conn,den.serialize())

    cross_tab_time=time.time()-start_time
    print("Cross tabulation time: ",cross_tab_time)

print("Total processing time: ", cross_tab_time + subquery_time + record_linkage_time)
conn.close()

