###############################################################
# 3. Data Preparation
###############################################################

df.shape

df.isnull().sum()

df = df[(df['Quantity'] > 0)]

df = df[~df["Invoice"].astype(str).str.contains("C", na=False)]

df.dropna(inplace=True)