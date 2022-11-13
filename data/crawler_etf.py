
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
    parser.add_argument('--year', type=int, default= 2022)
    args = parser.parse_args()
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
    for ticker in config['etf']: 
        print(f'{ticker}')
        df = yf.download(ticker, start = from_date, end = to_date)

        df.to_csv(BASEPATH+f'/data/rawdata/{ticker}_{args.year}.csv')

if __name__ =='__main__': 
    main()     
    

 