from recordlinkage.datasets import load_febrl4
import pandas as pd
import re
import numpy as np


dfA, dfB, true_links = load_febrl4(return_links=True)



print("Dataset A")
print(dfA.sort_index().head())
print("Dataset B")
print(dfB.sort_index().head())

def create_full_name(row):
    # Convert each part to string, replace NaN with empty string
    first_name = str(row['given_name']) if pd.notna(row['given_name']) else ''
    last_name = str(row['surname']) if pd.notna(row['surname']) else ''

    # Create the full name, skipping any empty parts
    full_name = ' '.join(filter(None, [first_name, last_name]))
    if len(full_name)<2:
        full_name='John Doe'
    return full_name

payload_var='Date of Birth'
cross_tab_var='state'
# Generate a dictionary mapping unique states to numerical values
#cross_tab_var_dict = {x: i+1 for i, x in enumerate(dfB[cross_tab_var].unique())}

# Map the states to numeric values
#dfB[cross_tab_var] = dfB[cross_tab_var].map(cross_tab_var_dict)

dfA['ID']=dfA.index
dfA['ID']=dfA['ID'].apply(lambda x: int(re.sub(r'\D','',x)))
#get age from date of birth
age_array=dfA['date_of_birth'].values
for i in range(len(age_array)):
  #print(age_array[i])
  try:
    age=age_array[i][0:4]
  except:
    age=0
  #print(age)
  age_array[i]=age
dfA['age']=age_array
dfA['age']=dfA['age'].apply(lambda x: int(x))
dfA['age']=dfA['age'].apply(lambda x:2024-x)

# join first and last name
dfA['Full Name'] = dfA.apply(create_full_name, axis=1)

dfB['ID']=dfB.index
dfB['ID']=dfB['ID'].apply(lambda x: int(re.sub(r'\D','',x[0:-1])))
#get age from date of birth
age_array=dfB['date_of_birth'].values
dfB['age']=age_array
for i in range(len(age_array)):
  #print(age_array[i])
  try:
    age=age_array[i][0:4]
  except:
    age=0
  #print(age)
  age_array[i]=age
dfB['age']=age_array
dfB['age']=dfB['age'].apply(lambda x: int(x))
dfB['age']=dfB['age'].apply(lambda x:2024-x)


#make an income column
dfB['income']=np.random.uniform(26000,110000,dfB.shape[0]).astype(int)

#discretize age into age ranges
dfB['age_band']=pd.qcut(dfB['age'],5)
tempB= pd.get_dummies(dfB['age_band'],prefix='income',dtype=int)


#make income based on age band columns
dfB=dfB.merge(tempB.multiply(dfB['income'],axis=0),left_index=True,right_index=True)

#join first and last name
dfB['Full Name'] = dfB.apply(create_full_name, axis=1)

df1 = dfA[['ID', 'Full Name', 'date_of_birth']]
df1.rename(columns={'Full Name Original': 'Fuzzy Full Name','date_of_birth':'Date of Birth'}, inplace=True)
df2 = dfB[['ID', 'Full Name', 'income','date_of_birth','state'] + dfB.columns[dfB.columns.str.startswith("income")].to_list()]

df2 = df2.merge(df1[['ID', 'Full Name']], on='ID', how='left', suffixes=('', ' Original'))

# Rename the merged 'Full Name' column to 'Original Full Name'
df2.rename(columns={'Full Name Original': 'Fuzzy Full Name','date_of_birth':'Year of Birth'}, inplace=True)


df2['Year of Birth'] = df2['Year of Birth'].apply(lambda x: int(x))
avg_dob = df2.loc[df2['Year of Birth']!=0, 'Year of Birth'].mean()

# Replace 0 with the computed average
df2['Year of Birth'] = df2['Year of Birth'].replace(0, int(avg_dob))  # Convert to integer for consistency
df2['age']=2024- df2['Year of Birth']

df1=df1.reset_index()
df2=df2.reset_index()


df1=df1[['ID','Full Name']]

df2=df2[['ID','Fuzzy Full Name','age','state']]

top_states = df2['state'].value_counts().nlargest(5).index

# Replace states that are not in the top 5 with "Other"
df2['state'] = df2['state'].apply(lambda x: x if x in top_states else 'other')


import csv
import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from random import randint
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import itertools
from sklearn.metrics import jaccard_score
from nltk import ngrams
from sklearn.preprocessing import OneHotEncoder
import string
import hashlib
import math
from numpy.linalg import norm
import time
import psutil
import pickle

df1=pd.read_csv('dataset/client_df.csv')
df2=pd.read_csv('dataset/server_df.csv')

#Parameters
shingle_size = 3
num_permutations = 200
num_permutations2 = 50
num_permutations3 = 100
max_hash = (2**20)-1 #(2 ** 31) - 1


# Function to generate the shingles of a string
def generate_shingles(string, shingle_size):
    shingles = set()
    for i in range(len(string) - shingle_size + 1):
        shingle = string[i:i + shingle_size]
        shingles.add(shingle)
    return shingles

# Function to generate a hash value for a shingle
def hash_shingle(shingle):
    return int(hashlib.sha256(shingle.encode()).hexdigest(), 32)

# Function to generate a random permutation function
def generate_permutation_function(num_permutations, max_hash):
    def permutation_function(x):
        random.seed(x)
        a = random.randint(1, max_hash)
        b = random.randint(0, max_hash)
        return lambda h: (a * h + b) % max_hash
    return [permutation_function(i) for i in range(num_permutations)]

# Function to compute the MinHash signature of a set of shingles
def compute_minhash_signature(shingles, permutation_functions):
    signature = [float('inf')] * len(permutation_functions)
    for shingle in shingles:
        shingle_hash = hash_shingle(shingle)
        for i, permutation in enumerate(permutation_functions):
            hashed_value = permutation(shingle_hash)
            if hashed_value < signature[i]:
                signature[i] = hashed_value
    return signature

permutation_functions = generate_permutation_function(num_permutations, max_hash)
permutation_functions2 = generate_permutation_function(num_permutations2, max_hash)
permutation_functions3 = generate_permutation_function(num_permutations3, max_hash)

df_names=df1

strings1 = df_names['Full Name']
shingles1 = [generate_shingles(string, shingle_size) for string in strings1]
#
signatures1 = [compute_minhash_signature(shingle, permutation_functions) for shingle in shingles1]
signatures12 = [compute_minhash_signature(shingle, permutation_functions2) for shingle in shingles1]
signatures13 = [compute_minhash_signature(shingle, permutation_functions3) for shingle in shingles1]
#
i=2
df_names.insert(i, 'Signature-200', signatures1)
signatures_at_responser = df_names['Signature-200'].to_numpy()
signatures_at_responser = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser])
df_names.insert(i+1, 'Signature_Norm-200', signatures_at_responser.tolist())
#
df_names.insert(i+2, 'Signature-50', signatures12)
signatures_at_responser2 = df_names['Signature-50'].to_numpy()
signatures_at_responser2 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser2])
df_names.insert(i+3, 'Signature_Norm-50', signatures_at_responser2.tolist())
#
df_names.insert(i+4, 'Signature-100', signatures13)
signatures_at_responser3 = df_names['Signature-100'].to_numpy()
signatures_at_responser3 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser3])
df_names.insert(i+5, 'Signature_Norm-100', signatures_at_responser3.tolist())

df_fuzzy_names=df2

strings1 = df_fuzzy_names['Fuzzy Full Name']
shingles1 = [generate_shingles(string, shingle_size) for string in strings1]
#
signatures1 = [compute_minhash_signature(shingle, permutation_functions) for shingle in shingles1]
signatures12 = [compute_minhash_signature(shingle, permutation_functions2) for shingle in shingles1]
signatures13 = [compute_minhash_signature(shingle, permutation_functions3) for shingle in shingles1]
#
df_fuzzy_names.insert(3, 'Fuzzy Signature-200', signatures1)
signatures_at_responser = df_fuzzy_names['Fuzzy Signature-200'].to_numpy()
signatures_at_responser = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser])
df_fuzzy_names.insert(4, 'Fuzzy Signature_Norm-200', signatures_at_responser.tolist())
#
df_fuzzy_names.insert(5, 'Fuzzy Signature-50', signatures12)
signatures_at_responser2 = df_fuzzy_names['Fuzzy Signature-50'].to_numpy()
signatures_at_responser2 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser2])
df_fuzzy_names.insert(6, 'Fuzzy Signature_Norm-50', signatures_at_responser2.tolist())
#
df_fuzzy_names.insert(7, 'Fuzzy Signature-100', signatures13)
signatures_at_responser3 = df_fuzzy_names['Fuzzy Signature-100'].to_numpy()
signatures_at_responser3 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser3])
df_fuzzy_names.insert(8, 'Fuzzy Signature_Norm-100', signatures_at_responser3.tolist())

exact_names_filename = f"client_names_5k_lsh200-50-100.pkl"
fuzzy_names_filename = f"server_fuzzy_names_5k_lsh200-50-100.pkl"

df_names.to_pickle('dataset/' + exact_names_filename)
df_fuzzy_names.to_pickle('dataset/' + fuzzy_names_filename)

#print(df_names.head())
#print(df_fuzzy_names.head())