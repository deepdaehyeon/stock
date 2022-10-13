import pandas as pd 
import investpy as iv
import investiny as it 
from datetime import datetime

ticker = 'QQQ'
types = 'ETF'
from_date = '01/01/2000'
to_date = datetime.now().strftime('%m/%d/%Y') 
 
id = it.search_assets(ticker, limit =1 , type = types)[0]['ticker']

dict = it.historical_data(id, from_date= from_date, to_date= to_date)
print(id)
df = pd.DataFrame.from_dict(dict)