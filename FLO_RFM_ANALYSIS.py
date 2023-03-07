import pandas as pd
import datetime as dt

###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this purpose, the behaviors of the customers will be defined and groups will be formed according to these behavior clusters.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases as OmniChannel
# (both online and offline shopper) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Last shopping date made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

###############################################################
# TASKS
###############################################################

# TASK 1: Data Understanding and Data Preparation
           # 1. Read flo_data_20K.csv
           # 2. Data sets:
                     # a. top 10 observations,
                     # b. variable names,
                     # c. descriptive statistics,
                     # d. null value,
                     # e. Variable types, review it.

df = pd.read_csv("datasets/flo_data_20k.csv")
df.head(10)
df.columns
df.shape
df.isnull().sum()
df.dtypes
df.describe().T


# 3.Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total purchases and spend.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Examine the variable types. Change the type of variables that express date to date.
#df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)

for i in df.columns:
    if "date" in i:
        df[i] = df[i].apply(pd.to_datetime)

[df[i].apply(pd.to_datetime) for i in df.columns if "date" in i]

#df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
#df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
#df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
#df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")

# 5. See the breakdown of the number of customers, average number of products purchased, and average spend across shopping channels.

df.groupby("order_channel").agg({"order_num_total": "mean",
                                 "customer_value_total": "mean"})

# 6. Rank the top 10 customers with the highest revenue.

df[["master_id", "customer_value_total"]].sort_values("customer_value_total", ascending=False).head(10)

# 7. List the top 10 customers with the most orders.

df["master_id"].sort_values("order_num_total", ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.



###############################################################
# TASK 2: Calculation RFM Metrics
###############################################################

# Analysis date 2 days after the last shopping date in the data set

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)

# customer_id, recency, frequency ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                             "order_num_total": lambda x: x,
                             "customer_value_total": lambda x: x})

rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]

###############################################################
# TASK 3: Calculating RF and RFM Scores
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

###############################################################
# TASK 4: Definition of RF Scores as Segments
###############################################################

# Segment definition and converting RF_SCORE to segments with the help of defined seg_map so that the generated RFM scores can be explained more clearly.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

###############################################################
# TASK 5: Action time!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
#olarak kaydediniz.

rfm2 = pd.merge(rfm, df[['interested_in_categories_12','master_id']],on='master_id', how='left')

new_df = pd.DataFrame()
new_df["customer_id"] = rfm2[(rfm2["segment"] == "loyal_customers") & (rfm2["interested_in_categories_12"].str.contains("KADIN"))].index

new_df.to_csv("yeni_marka_hedef_musteri_id")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
#olarak kaydediniz.

new_df2 = pd.DataFrame()

new_df2["customer_id"] = rfm2[((rfm2["segment"] == "cant_loose") | (rfm2["segment"] == "new_customers")) & ((rfm2["interested_in_categories_12"].str.contains("ERKEK")) | (rfm2["interested_in_categories_12"].str.contains("COCUK") == True))].index

new_df2.to_csv("indirim_hedef_müşteri_ids")