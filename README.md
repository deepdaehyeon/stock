# Nuts 

This is Nasdaq aUto Trading System developed by dh.kim
the system is based on reinforcement learning 
we use collab for model training 

-----------
## Tools 
* trading env(gym) : https://github.com/notadamking/Stock-Trading-Environment.git

* rl lib(ray) : https://github.com/ray-project/ray.git 

* Crawling: yahoo-finance 

* Trading platform: Hantoo (web socket)

-----------
## Data

* Strategy: 
focus on macro economic factor, trade TQQQ & SQQQ etfs   

* Features(x)
> * Nasdaq 100 
> * Energy: WTI, gas .. 
> * Commodities: gold, copper, 
> * Bond (2, 5, 10 year)
> * usdIndex 
> * VIX index
> * NFPs, CPI, GDP 
> * cripto (bitcoin)
> * Events: PPI, CPI, 잭슨홀 미팅 발표전일 -> Boolean
> * Day: mon, tue, wed, thd, fri 

* Preprocess 
> * open, close, high, low: scaled by open 
> * volumn: scaled by stdv scaler 
> * MA(close, volumn moving average) 5 20 200 day: scaled by open, stdv   
> * 전고점, 전저점: scaled by open 
  

* Action space 
> * TBD



