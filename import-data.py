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

df1['Full Name'] = df1['Full Name'].str.lower()
#df2['Full Name'] = df2['Full Name'].str.lower()
df2['Fuzzy Full Name'] = df2['Fuzzy Full Name'].str.lower()

df1.to_csv('dataset/client_df.csv')
df2.to_csv('dataset/server_df.csv')