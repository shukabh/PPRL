import socket
from helpers import *
import tenseal as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# -------------------------------
# Parameters
# -------------------------------
NO_OF_QUERIES = 3000

# -------------------------------
# Step 0: Load and preprocess client dataset
# -------------------------------
dataset = pd.read_pickle('dataset/df_names_10k_lsh200-50-100.pkl')
print("Dataset loaded")

# Keep only relevant columns
dataset = dataset[['ID', 'Signature_Norm-200', 'Signature_Norm-50']]

# Normalize the 200-dimensional signature vectors
scaler = StandardScaler()
signatures_200 = np.array(dataset['Signature_Norm-200'].tolist())
scaled_signatures = scaler.fit_transform(signatures_200)

# Extract query IDs and the 50-dimensional signature vectors
query_ids = dataset['ID'].to_numpy()
signatures_50 = np.array(dataset['Signature_Norm-50'].tolist())

# -------------------------------
# Step 1: Set up secure connection to the server
# -------------------------------
client_socket = socket.socket()
client_socket.connect(('localhost', 1234))
print("Connected to server.")

#change code to receive no of clusters from server

no_of_clusters = recv_data(client_socket)
# -------------------------------
# Step 2: Create TenSEAL context and share public part with server
# -------------------------------
context = create_context()
send_data(client_socket, context.serialize())
print("Sent public context to server.")

# -------------------------------
# Step 3: Encrypt query vectors and send to server
# -------------------------------
print('Record linkge protocol started...')
enc_querier = ts.ckks_tensor(context, signatures_50[:NO_OF_QUERIES], None, True)
enc_querier_scaled = ts.ckks_tensor(context, scaled_signatures[:NO_OF_QUERIES], None, True)
send_data(client_socket, enc_querier_scaled.serialize())
#print("Sent scaled encrypted queries.")

# -------------------------------
# Step 4: Receive encrypted similarity results from server
# -------------------------------
enc_result = recv_data(client_socket)
enc_querier_scaled = ts.ckks_tensor_from(context, enc_result)
dec_cos_sim = enc_querier_scaled.decrypt().tolist()

# Identify most matching cluster for each query
most_matching_cluster = [np.argmax(row) for row in dec_cos_sim]

# -------------------------------
# Step 5: Send one-hot encoded cluster matches to server
# -------------------------------
sign_centroids = np.zeros((NO_OF_QUERIES, no_of_clusters))
for i, idx in enumerate(most_matching_cluster):
    sign_centroids[i][idx] = 1

enc_sign_centroids = ts.ckks_tensor(context, sign_centroids, None, True)
send_data(client_socket, enc_sign_centroids.serialize())
#print("Sent encoded centroid matches.")

# -------------------------------
# Step 6: Receive maximum cluster size and send encrypted query again
# -------------------------------
max_cluster_size = recv_data(client_socket)
#print('Max cluster size:', max_cluster_size)
send_data(client_socket, enc_querier.serialize())
#print("Resent encrypted query to server.")

# -------------------------------
# Step 7: Receive encrypted record linkage scores
# -------------------------------
res = []
for _ in tqdm(range(max_cluster_size)):
    enc_cs_bytes = recv_data(client_socket)
    enc_cs = ts.ckks_tensor_from(context, enc_cs_bytes)
    res.append(enc_cs.decrypt().tolist())

res = np.array(res).T
res[res > 1.01] = 0  # Threshold cleanup

print("Subquery protocol started...")
# -------------------------------
# Step 8: Build binary payload vector from record linkage result
# -------------------------------
cluster_ids = recv_data(client_socket)
id_candidates = [cluster_ids[most_matching_cluster[i]][np.argmax(res[i])] for i in range(NO_OF_QUERIES)]
payload_mask = np.isin(cluster_ids, id_candidates).astype(int)

# Send encrypted payload vectors to server
for i in range(max_cluster_size):
    enc_vec = ts.ckks_tensor(context, payload_mask[:, i])
    send_data(client_socket, enc_vec.serialize())

#print("Sent encrypted payload mask.")

# -------------------------------
# Step 9: Receive and decrypt sub-query sizes
# -------------------------------
dec_subquery_sizes = []
n = recv_data(client_socket)

for _ in range(n):
    vec_bytes = recv_data(client_socket)
    vec = ts.ckks_vector_from(context, vec_bytes)
    dec_subquery_sizes.append(vec.decrypt()[0])

# Send decrypted sub-query sizes back to server
send_data(client_socket, dec_subquery_sizes)

# -------------------------------
# Step 10: Receive final ratios from server
# -------------------------------
print("Main computation started...")
keys, ratios = [], []
num_keys = recv_data(client_socket)

for _ in range(num_keys):
    key = str(recv_data(client_socket))
    num = ts.ckks_vector_from(context, recv_data(client_socket)).decrypt()[0]
    den = ts.ckks_vector_from(context, recv_data(client_socket)).decrypt()[0]
    ratio = num / den if den != 0 else 0
    keys.append(key)
    ratios.append(ratio)

# -------------------------------
# Step 11: Output results in a DataFrame
# -------------------------------
df_ratios = pd.DataFrame({'key': keys, 'average': ratios})
print(df_ratios)

# -------------------------------
# Step 12: Close the socket
# -------------------------------
client_socket.close()

'''
no_of_queries = 100
no_of_clusters = 50
#Load client dataset
print("Started")
dataset1 = pd.read_pickle('dataset/df_names_10k_lsh200-50-100.pkl')
print("dataset 1 loaded")
dataset1 = dataset1[['ID', 'Signature_Norm-200', 'Signature_Norm-50']]


scaler = StandardScaler()
normalised_signatures = dataset1['Signature_Norm-200']
normalised_signatures = np.array([x for x in normalised_signatures.to_numpy()])
normalised_signatures_scaled = scaler.fit_transform(normalised_signatures)

query_IDs = dataset1['ID']
query_IDs = np.array([x for x in query_IDs.to_numpy()])

normalised_signatures_50 = dataset1['Signature_Norm-50']
normalised_signatures_50 = np.array([x for x in normalised_signatures_50.to_numpy()])


#network
client_socket = socket.socket()
client_socket.connect(('localhost', 12345))
print("Connected to server.")

# Step 1: Create context and send public part
context = create_context()
public_context_bytes = context.serialize()
send_data(client_socket, public_context_bytes)
print("Sent public context to server.")

querier_IDs = query_IDs[0:no_of_queries]
enc_querier = ts.ckks_tensor(context, normalised_signatures_50[0:no_of_queries], None, True)
enc_querier_scaled = ts.ckks_tensor(context, normalised_signatures_scaled[0:no_of_queries], None, True)


#writeCkks(enc_querier.serialize(), "out/enc_querier")
#writeCkks(enc_querier_scaled.serialize(), "out/enc_querier_scaled")

send_data(client_socket,enc_querier_scaled.serialize())
print("Sent enc_querier_scaled to server.")




# Step 2: Wait for encrypted result from server
encrypted_result_bytes = recv_data(client_socket)
print("Recieved enc_querier")
enc_querier_scaled = ts.ckks_tensor_from(context, encrypted_result_bytes)

#print(enc_querier_scaled.shape)
#readCkks("out/enc_querier_scaled")

# Step 3: Decrypt
dec_cos_sim_with_centroids=enc_querier_scaled.decrypt().tolist()


most_matching_cluster = []
for i in range(no_of_queries):
    most_matching_cluster.append(np.argmax(dec_cos_sim_with_centroids[i]))

#print("Most matching cluster ", most_matching_cluster)


sign_of_centroids = np.zeros((no_of_queries, no_of_clusters))
for i in range(no_of_queries):
    sign_of_centroids[i][most_matching_cluster[i]] = 1

enc_sign_of_centroids = ts.ckks_tensor(context, sign_of_centroids, None, True)
#writeCkks(enc_sign_of_centroids.serialize(), "out/enc_sign_of_centroids")
send_data(client_socket,enc_sign_of_centroids.serialize())
print("Sent sign of centroids")

#step 2
max_cluster_size=recv_data(client_socket)
print('Max cluster size',max_cluster_size)



send_data(client_socket,enc_querier.serialize())
print("Sent enc_querier to server.")

res = []
for i in tqdm(range(max_cluster_size)):
    
    #writeCkks(enc_sign_of_centroids_tmp.serialize(), "out/cs")
    #print(time.time() - inner_time)
    enc_sign_of_centroids_tmp_bytes=recv_data(client_socket)
    enc_sign_of_centroids_tmp=ts.ckks_tensor_from(context, enc_sign_of_centroids_tmp_bytes)
    cs = enc_sign_of_centroids_tmp.decrypt().tolist()
    res.append(cs)
    #print(i, " CS: ", cs)

#record_linkage_time=time.time() - start_time
#print(record_linkage_time)  

res = np.array(res).transpose()
res[res > 1.01] = 0

cluster_dataset2_IDs=recv_data(client_socket)

ID_listA=[cluster_dataset2_IDs[most_matching_cluster[i]][np.argmax(res[i])] for i in range(no_of_queries)]

# Convert ID_listA to a numpy array for efficient comparison
ID_listA_set = set(ID_listA)

# Use broadcasting and vectorized operations to create the output array
payload_querier = np.isin(cluster_dataset2_IDs, list(ID_listA_set)).astype(int)


for i in range(max_cluster_size):
    enc_querier_vec=ts.ckks_tensor(context,payload_querier[:,i])
    send_data(client_socket, enc_querier_vec.serialize())

print("Sent payload_querier")

dec_sub_query_sizes=[]
n=recv_data(client_socket)

for i in range(n):
    x_bytes=recv_data(client_socket)
    x=ts.ckks_vector_from(context,x_bytes)
    #print(x)
    #print(x.decrypt())
    dec_sub_query_sizes.append(x.decrypt()[0])

#print("Sub query sizes",dec_sub_query_sizes)

send_data(client_socket,dec_sub_query_sizes)


# Assume conn is the socket connection to server
# Assume context is a ts.Context with decryption capability (has secret key)

keys = []
ratios = []

number_of_keys = recv_data(client_socket)

for _ in range(number_of_keys):  # replace with actual number of keys expected
    # Receive key (string)
    key = recv_data(client_socket)
    key = str(key)
    # Receive numerator
    num_bytes = recv_data(client_socket)
    num = ts.ckks_vector_from(context, num_bytes)
    decrypted_num = num.decrypt()[0]  # Assuming scalar vector

    # Receive denominator
    den_bytes = recv_data(client_socket)
    den = ts.ckks_vector_from(context, den_bytes)
    decrypted_den = den.decrypt()[0]  # Assuming scalar vector

    # Compute ratio (avoid div-by-zero)
    ratio = decrypted_num / decrypted_den if decrypted_den != 0 else 0

    # Store results
    keys.append(key)
    ratios.append(ratio)

# Create DataFrame
df_ratios = pd.DataFrame({
    'key': keys,
    'value': ratios
})

print(df_ratios)

client_socket.close()
'''