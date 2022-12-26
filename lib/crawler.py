from datetime import datetime
import yaml
import os
import yfinance as yf 
from dataclasses import dataclass
from typing import Dict, List

# Modules
@dataclass
class Crawler:
    export_dir: str = 'data/rawdata'
    year: str = '2000' 
    month: str = '01' 
    day: str = '01'
    
    def __post_init__(self): 
        self.BASEPATH = os.getcwd().replace('\\', '/')
        with open(f'{self.BASEPATH}/env/config.yaml') as conf:
            config = yaml.load(conf, Loader=yaml.FullLoader)
        self.config =  config 
        
    def run(self, etf_list = None, index_list = None):
        from_date = f'{self.year}-{self.month}-{self.day}'
        to_date = datetime.now().strftime('%Y-%m-%d') 
        
        # ETF Crawling
        if etf_list is None:
            etf_list = self.config['etf']
        for ticker in etf_list:
            df = yf.download(ticker, start=from_date, end=to_date)
            df.to_csv(f'{self.BASEPATH}/{self.export_dir}/{ticker}.csv')
        
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
    

 