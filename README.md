# Nuts 
This is Nasdaq aUto Trading System developed by dh.kim 
-----------
* Trading platform: Hantoo (web socket)

* Crawling: investpy & investiny(temporally use)

* Strategy: 
focus on macro economic factor, trade TQQQ & SQQQ   

* Features(x)
> * Nasdaq 100 
> * Energy: WTI, gas .. 
> * Commodities: gold, copper, 
> * Bond (2, 5, 10 year)
> * usdIndex 
> * cripto (bitcoin)
> * Events: PPI, CPI, 잭슨홀 미팅 발표전일 -> Boolean

* Preprocess : all features scaled by previous open price
> * open, close, high, low, volumn 
> * MA(moving average) 
> * 전고점, 전저점 

* Target (y)
> Nasdaq 100 average price for next 10 days 

* XAI: uncertainty (additional)
> use ensemble method 

------------
## Models 