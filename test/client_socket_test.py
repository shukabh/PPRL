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
NO_OF_QUERIES = 5000
THRESHOLD = 0.9
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
query_IDs = dataset['ID'].to_numpy()
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

cluster_ids = recv_data(client_socket)
id_candidates = [cluster_ids[most_matching_cluster[i]][np.argmax(res[i])] for i in range(NO_OF_QUERIES)]

fuzzy_query_IDs = recv_data(client_socket)
querier_IDs = query_IDs[0:NO_OF_QUERIES]

#print(id_candidates)
'''
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(NO_OF_QUERIES):
    most_matching_cs_index = np.argmax(res[i])
    if res[i][most_matching_cs_index] > THRESHOLD: #checks if RL sim is above threshold contributes to TP or FP
        if querier_IDs[i] == id_candidates[i]:
            TP += 1
        else:
            FP += 1
    else: #if score is less than threshold contributes to TN or FN
        if querier_IDs[i] in fuzzy_query_IDs:
            FN += 1
        else:
            TN += 1
'''


TP = 0
FP = 0
TN = 0
FN = 0

for i in range(NO_OF_QUERIES):
    most_matching_cs_index = np.argmax(res[i])
    similarity_score = res[i][most_matching_cs_index]
    predicted_match = similarity_score > THRESHOLD
    true_match = querier_IDs[i] == id_candidates[i]

    if predicted_match:
        if true_match:
            TP += 1
        else:
            FP += 1
    else:
        if true_match:
            FN += 1
        else:
            TN += 1

# Precision, Recall, F1-Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Output
print("Threshold: ", THRESHOLD)
print("No of queries: ",NO_OF_QUERIES)
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"True Negatives: {TN}")
print(f"False Negatives: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")





client_socket.close()
