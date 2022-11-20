from datetime import datetime
import yaml
import os
import yfinance as yf 

# Modules
class Crawler:
    def __init__(self, args, export_dir='data/rawdata'):
        self.BASEPATH = os.getcwd().replace('\\', '/')
        self.export_dir = export_dir
        self.args = args 
        with open(f'{self.BASEPATH}/config/config.yaml') as conf:
            config = yaml.load(conf, Loader=yaml.FullLoader)
        self.config =  config 
        
    def crawl(self, ticker_list = None):
        from_date = f'{self.args.year}-{self.args.month}-{self.args.day}'
        to_date = datetime.now().strftime('%Y-%m-%d') 
        
        # ETF Crawling
        if ticker_list is None:
            ticker_list = self.config['etf']
            for ticker in ticker_list:
                df = yf.download(ticker, start=from_date, end=to_date)
                df.to_csv(f'{self.BASEPATH}/{self.export_dir}/{ticker}.csv')
        
        # Economic indicator 
        elif ticker_list == 'cpi': 
            pass 
        
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
    

 