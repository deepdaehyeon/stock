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

# Libraries
if True: 
    from datetime import datetime
    import yaml
    import os
    import argparse 
    import yfinance as yf 

# Config & Args 
if True: 
    # Argparser 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--year', type=int, default= 2022)
    args = parser.parse_args()
    
    # Configs
    BASEPATH =os.getcwd() +'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 

# Main
def main(): 
    # Set date 
    from_date = f'{args.year}-01-01'
    if args.year == 2022: 
        to_date = datetime.now().strftime('%Y-%m-%d') 
    else: 
        to_date = f'{args.year}-12-31'
    # Data crawling
    # TBD 

if __name__ =='__main__': 
    main()     
    

 