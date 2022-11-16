
# Libraries
if True: 
    from datetime import datetime
    import yaml
    import os
    import argparse 
    import yfinance as yf 

# Config & Args 
if True: 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--year', type=str, default= '2022')
    parser.add_argument('--month', type=str, default= '01')
    parser.add_argument('--day', type=str, default= '01')
    args = parser.parse_args()
    BASEPATH =os.getcwd() +'/..' 
    with open(BASEPATH+'/config/config.yaml') as conf:
        config = yaml.load(conf, Loader = yaml.FullLoader) 

# Main
def main(args): 
    # Set date 
    from_date = f'{args.year}-{args.month}-{args.day}'
    to_date = datetime.now().strftime('%Y-%m-%d') 

    # Data crawling
    for ticker in config['etf']: 
        print(f'{ticker}')
        df = yf.download(ticker, start = from_date, end = to_date)
        df.to_csv(BASEPATH+f'/data/rawdata/{ticker}.csv')

if __name__ =='__main__': 
    main(args)     
    

 