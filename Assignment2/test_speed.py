
import numpy as np
import pandas as pd

# df = pd.read_csv("data/fake_data/training_fake.csv")

clean = "test"

df = pd.read_csv(f"data/{clean}_set_VU_DM.csv")
df['n_srchitems'] = df.groupby('srch_id')['srch_id'].transform('count')
df['n_booked'] = df.groupby('prop_id')['prop_id'].transform('count')
df["srch_rank"] = df.groupby("srch_id")["srch_id"].rank("first", ascending=True)
# print(df[["srch_id", "prop_id", "n_srchitems","n_booked", "srch_rank"]].to_string())

df.to_csv(f"data/1_BIG_{clean}.csv")
print(df.shape)
