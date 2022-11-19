from datetime import datetime
import os
import yaml
import yfinance as yf


class EtfCrawler:
    def __init__(self, export_dir='data/rawdata'):
        self.BASEPATH = os.getcwd().replace('\\', '/')
        self.export_dir = export_dir

    def crawl(self, ticker_list: list = None, from_year: int = 2000, to_year: int = 2022):
        if ticker_list is None:
            with open(f'{self.BASEPATH}/config/config.yaml') as conf:
                config = yaml.load(conf, Loader=yaml.FullLoader)
                ticker_list = config['etf']

        for year in range(from_year, to_year+1):
            # Set date
            from_date = f'{year}-01-01'
            if year == int(datetime.now().strftime('%Y')):
                to_date = datetime.now().strftime('%Y-%m-%d')
            else:
                to_date = f'{year}-12-31'

            # Data Crawling
            for ticker in ticker_list:
                df = yf.download(ticker, start=from_date, end=to_date)
                df.to_csv(f'{self.BASEPATH}/{self.export_dir}/{ticker}_{year}.csv')
    

 