import numpy as np
import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'TsHzstzW28XTsmBjJZug'
quandl.ApiConfig.verify_ssl = False

# CALL DATA (WTI CRUDE OIL PRICE)
returns = quandl.get("EIA/PET_RWTC_D",
                     start_date="2019-01-01", end_date="2019-06-10",
                     collapse="daily",
                     returns=np
                     # returns=pd.DataFrame,

                     # return values transformation="rdiff"
                     # none

                     )
print(returns)


# Download Dataset
# quandl.bulkdownload("EIA/PET_RWTC_D")

# CALL DATA (MER/F1 is Mergent Global Fundamentals Dataset, for Nokia and DB)
mf1 = quandl.get_table('MER/F1',
                       compnumber=["39102", "2438"],
                       # to download individual columns use
                       #qopts={"columns": ['compnumber', 'ticker', 'reportdate', 'indicator', 'amount']},

                       # True = 1MIL Rows
                       paginate=True)

print(mf1)


# CLOSING PRICES
clsprice = quandl.get_table('WIKI/PRICES',
                            qopts={'columns': ['ticker', 'date', 'close']},
                            ticker=['AAPL', 'MSFT'],
                            date={'gte': '2016-01-01', 'lte': '2017-01-01'})

print(clsprice)


'''
transformations:
diff
rdiff
rdiff_from
cumul
normalize

compnumber samples [2010-12-31 - 2015-12-31]:
83356
71388
115788
86699
74463
17630
76258
85610
111113
2438
12760
14267
39102
99247
15544
101084
86867
15845
104579
15867
107255
1409
109135
73128
131058
52460
372
6139
12161
133772
'''
