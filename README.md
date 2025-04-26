# Input Output Privacy Record Linkage Framework

This project involves two parties: a **client** and a **server**, each possessing a dataset they are unwilling to share with one another. Despite this, the client needs to perform **record linkage** with the server's dataset and generate summary statistics from relevant columns of the server's data.

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

