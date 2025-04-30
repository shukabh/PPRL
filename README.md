# Input Output Privacy Record Linkage Framework

This project involves two parties: a **client** and a **server**, each possessing a dataset they are unwilling to share with one another. Despite this, the client needs to perform **record linkage** with the server's dataset and generate summary statistics from relevant columns of the server's data.

More generally this repository outlines a privacy-preserving system for performing secure, encrypted record linkage between client and server datasets while ensuring both confidentiality and statistical integrity and allowing collaborative data analysis.

## Example Scenario

Consider the following example: 

- The **client** is a researcher with a dataset containing names of HIV-infected patients from a specific county.
- The **server** is an insurance agency holding a dataset with information about all citizens in the county.

The researcher wants to create a contingency table showing the **average income** of HIV-infected patients by ethnicity. Using this framework of **homomorphic encryption**, the researcher can perform record linkage with the insurance agency's dataset and generate an **encrypted 2-way table**, which only the researcher can decrypt.

## Input Privacy
The server cannot see the dataset of the client and cannot even know how many matches are made from its dataset. 

## Output Privacy
The client cannot see the attributes of single individuals or a very small number of individuals. The results are summary statistics and should not reveal any personal information.

## Technique and threat model
Input privacy is achieved by employing homomorphic encryption. For output privacy as there is a possiblity of client being malicous to extract individual values the server devises a subquery size protocol which makes sure that client's queries if they turn out to be smaller than a threshold will be rejected by the server. 

## Project Example

In this repository, we demonstrate the framework using a sample of **5000 residents in Australia**, linking the dataset to itself and calculating the **average age** of residents by province.

## Getting Started

To get started, follow these steps:
Start a virtual enviroment

```bash
python -m venv venv && source venv/bin/activate
```

### 1. Install Dependencies  

Install the necessary packages using the following command:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python server_socket.py
```
### 3. Start the Client 
```bash
python client_socket.py
```
On the server side we will see

```bash
Server listening...
Client connected: ('127.0.0.1', 63763)
Record linkage protocol started...
100%|███████████████████████| 192/192 [09:30<00:00,  2.97s/it]
Record linkage time: 607.83 seconds
Subquery protocol started...
Subquery protocol time: 238.71 seconds
Subquery protocol passed
Main computation started...
Cross tabulation time: 231.48 seconds
Total processing time: 1078.02 seconds
```

On the client side we will see
```bash
Dataset loaded
Connected to server.
Sent public context to server.
Record linkage protocol started...
100%|███████████████████████| 192/192 [09:31<00:00,  2.98s/it]
Subquery protocol started...
Main computation started...
     key    average
0    vic  76.231863
1  other  76.233717
2    nsw  75.473042
3    qld  75.445471
4     sa  73.142180
5     wa  76.000000
```



## 📋 Process Overview

### 1. Data Import
Both the client and server import their respective datasets.

- **Server dataset must include**:
  - `matching_variable` (e.g., name or ID),
  - `payload_variable` (data of interest),
  - `cross_tab_variable` (for grouped analysis).

- **Client dataset must include**:
  - Only the `matching_variable`.

---

### 2. Data Preparation

- Matching variables (strings) are transformed into **n-grams**.
- **Locality-Sensitive Hashing (LSH)** is applied to convert these into numerical representations, preserving similarity.
- The choice of LSH algorithm depends on the required trade-off between precision and computational efficiency.

---

### 3. Clustering

Performed by the **server** to optimize matching speed:

- Records are grouped into a predefined number of clusters (`NO_OF_CLUSTERS`) based on the `matching_variable`.
- Each cluster has a size up to `MAX_CLUSTER_SIZE`; smaller clusters are padded with `NA` to ensure uniformity.

---

### 4. Matching / Record Linkage

- Records are matched based on their hashed values.
- A **similarity threshold** determines what constitutes a match:
  - Higher thresholds reduce false positives but may increase false negatives.
  - Lower thresholds are more permissive, allowing more matches but with increased risk of false positives.

---

### 5. Subquery Protocol

Ensures **client honesty** and prevents unauthorized access to individual records:

- The server computes intersection sizes for matches across cross-tabulated groups.
- These values are **obfuscated** by applying random multipliers and adding dummy data.
- The client decrypts the values, unaware of their significance.
- If the size of subqueries exceeds the `MIN_QUERY_SIZE` set by the server, the protocol proceeds to the next phase.

---

### 6. Tabulation Protocol

Final aggregation of matched data using encrypted computation:

- The server performs a **secure dot product** between matched records and cross-tabulated variables.
- Both numerator (sum) and denominator (count) are obfuscated using the same random multiplier.
- The client receives these values and computes **group-level averages** by dividing the obfuscated totals.

---

## 🛡️ Key Features

- **Privacy-Preserving**: Matching and aggregation are performed without exposing raw data.
- **Encrypted Communication**: Ensures that data is secure throughout the process.
- **Fuzzy Matching**: Supports different levels of precision via configurable thresholds and LSH methods.
- **Integrity Check**: Subquery protocol prevents misuse or targeted data extraction by the client.

---

## 🔧 Configuration Parameters

| Parameter         | Description                                     |
|------------------|-------------------------------------------------|
| `NO_OF_CLUSTERS` | Number of clusters used for preprocessing       |
| `MIN_QUERY_SIZE` | Minimum number of matches required per subquery |
| `threshold`      | Similarity threshold for record matching         |

---

## 📌 Use Cases

- Collaborative research using sensitive datasets
- Secure identity matching across institutions
- Federated analytics with healthcare or finance data

---

## 📎 License

MIT License. See [LICENSE](./LICENSE) for details.


