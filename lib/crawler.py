from datetime import datetime
import yaml
import os
import yfinance as yf 
from dataclasses import dataclass, field
from typing import Dict, List
from env.config import Config 

C = Config 

# Modules
@dataclass
class Crawler:
    from_date: str 
    to_date: datetime = field(default_factory = lambda: datetime.now().strftime('%Y-%m-%d'))  

    def __post_init__(self): 
        assert '-' in self.from_date, 'date format is 20xx-xx-xx' 
        assert '-' in self.to_date, 'date format is 20xx-xx-xx' 
        
    def run_etf(self, ticker_list):
        for ticker in ticker_list:
            df = yf.download(ticker, start=self.from_date, end=self.to_date)
            df.to_csv(os.path.join(C.datapath, 'rawdata', ticker+'.csv'))

    def run_indicator(self, indicator_list): 
        NotImplemented() 
        # Economic indicator 
        # if index_list is None: 
        #     ticker_list = self.config['index']
        
        '''
        macro economic indicator 의 경우, 
        아래의 사이트 중에서 직접 크롤링을 통해 구해야해서 좀 까다로움 
        Phase 2. 에서 진행하는게 나을듯  

        |Crawling libraries| 
        - beautifulsoup 
        - Selenium
        - scrapy
        https://www.projectpro.io/article/python-libraries-for-web-scraping/625 의 장단점 참고할 것.   

        |Data source|
        https://datasetsearch.research.google.com/ 를 통해서 데이터셋 소스를 찾을 수 있음 
        - https://fred.stlouisfed.org/series/T10Y2Y
        - www.currentmarketevaluation.com 
        '''
    

 