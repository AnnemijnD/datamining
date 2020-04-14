import pandas as pd
import numpy as np
import dateparser
from datetime import datetime


df = pd.read_excel('data/ODI-2020_cleaned.xlsx')
print(df['You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? '])
print(df.columns)

counter = 0
datelist = []
for index, row in df.iterrows():
    date = str(row['When is your birthday (date)?'])
    try:
        # print(row['When is your birthday (date)?'], "|||", dateparser.parse(row['When is your birthday (date)?']))
        if dateparser.parse(date) is not None:
            datelist.append(dateparser.parse(date))

            if date == "14 March":
                print(dateparser.parse(date))
    except:
        print("kon nie", date)
        print(index)
        print(type(date))
        counter +=1

print(counter)

for index, row in df.iterrows():
    row["hoi"]
    
# print(datelist)
# print(max(datelist)) # oldest
# print(min(datelist)) # earliest
