import argparse 
from utils.crawler import Crawler

# Args 
if True: 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--ticker', type=str, default= None)
    parser.add_argument('--year', type=str, default= '2000')
    parser.add_argument('--month', type=str, default= '01')
    parser.add_argument('--day', type=str, default= '01')
    args = parser.parse_args()

if __name__ =='__main__': 
    crawler = Crawler(args)
    crawler.crawl(args.ticker)
    

 