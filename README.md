# Secure Record Linkage Framework

This project involves two parties: a **client** and a **server**, each possessing a dataset they are unwilling to share with one another. Despite this, the client needs to perform **record linkage** with the server's dataset and generate summary statistics from relevant columns of the server's data.

## Example Scenario

Consider the following example: 

- The **client** is a researcher with a dataset containing names of HIV-infected patients from a specific county.
- The **server** is an insurance agency holding a dataset with information about all citizens in the county.

The researcher wants to create a contingency table showing the **average income** of HIV-infected patients by ethnicity. Using this framework of **homomorphic encryption**, the researcher can perform record linkage with the insurance agency's dataset and generate an **encrypted 2-way table**, which only the researcher can decrypt.

## Project Example

In this repository, we demonstrate the framework using a sample of **5000 residents in Australia**, linking the dataset to itself and calculating the **average age** of residents by province.

## Getting Started

To get started, follow these steps:

### 1. Install Dependencies  

Install the necessary packages using the following command:
```bash
pip install -r requirements.txt
