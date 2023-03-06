###############################################################
# 5. Calculating RFM Scores
###############################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"]

rfm[rfm["RFM_SCORE"] == "11"]