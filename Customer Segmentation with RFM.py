# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Functionalization of the Whole Process

###############################################################
# 1. Business Problem
###############################################################

# An e-commerce company wants to segment its customers and determine
# marketing strategies according to these segments.

# Dataset Story
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The dataset named Online Retail II includes the sales
# of an UK-based online store between 01122009 - 09122011.

# Variables
#
# InvoiceNo: Invoice number. The unique number of each transaction, namely the invoice.
# Aborted operation if it starts with C.
# StockCode: Product code. Unique number for each product.
# Description: Product name.
# Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in pounds sterling)
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.

###############################################################
# 2. Data Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_ = pd.read_excel(r"C:\Users\merve\Desktop\Bootcamp_miuul\CRM_Analytics\crmAnalytics\datasets\online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

## What is the number of unique items?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].unique()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

