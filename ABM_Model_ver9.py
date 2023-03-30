import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LimitOrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.stock_price = 10
        self.count = 0
        self.stepid = 0
        self.Close = [10] # 多加了一個紀錄收盤價的list
        
        
    def insert_order(self, order):
        """
        Insert an order into the appropriate bids or asks list and sort the list by price.
        """
        if order.type == 'buy':
            self.bids.append(order)
            self.bids.sort(key=lambda x: x.price, reverse=True)
        elif order.type == 'sell':
            self.asks.append(order)
            self.asks.sort(key=lambda x: x.price)
        
    def match_orders(self):
        """
        Match buy and sell orders in an order book by first-in first-out (FIFO)
        """
        self.count = 0 # Keep track of the number of successful transactions during the match_orders method execution
       
        transaction_price = self.stock_price
        tranList = []
        while self.bids and self.asks:
            for i in self.bids:
                if self.stepid - i.stepid == 15:
                    if i.who <= 180:
                        for x in range(1,181):
                            if lob_agent.trader_info[x]== i.who:
                                lob_agent.trader_info[x]['money'] += i.price
                        
                    if i.who > 180 and i.who <= 200:
                        for y in range(181,201):
                            if lob_agent.c.trader_info[y]== i.who:
                                lob_agent.c.trader_info[y]['money'] += i.price
                    
                    if i.who == 201:
                        for z in range(201,202):
                            if lob_agent.b.trader_info[z] == i.who:
                                lob_agent.b.trader_info[z]['money'] += i.price
                        
                    self.bids.remove(i)
                    
            for j in self.asks:
                if self.stepid - j.stepid == 15:
                    self.asks.remove(j)
            if self.bids[0].price >= self.asks[0].price:
                # Determine the transaction price as the average of the best bid and ask prices
                transaction_price = (self.bids[0].price + self.asks[0].price) / 2

                
                if self.bids[0].who <= 180:
                ## random trader股票數與金錢改變
                    lob_agent.trader_info[self.bids[0].who]['shares'] += 1
                    lob_agent.trader_info[self.bids[0].who]['money'] += (self.bids[0].price - transaction_price)
                if self.asks[0].who <= 180:
                    lob_agent.trader_info[self.asks[0].who]['money'] += transaction_price
                    
                if self.bids[0].who > 180 and self.bids[0].who <= 200:
                ## 圖表派股票數與金錢改變
                    lob_agent.c.trader_info[self.bids[0].who]['shares'] += 1
                    lob_agent.c.trader_info[self.bids[0].who]['money'] += (self.bids[0].price - transaction_price)
                if self.asks[0].who > 180 and self.asks[0].who <= 200:
                    lob_agent.c.trader_info[self.asks[0].who]['money'] += transaction_price
                
                if self.bids[0].who == 201:
                ## 大財團股票數與金錢改變
                    lob_agent.b.trader_info[self.bids[0].who]['shares'] += 1
                    lob_agent.b.trader_info[self.bids[0].who]['money'] += (self.bids[0].price - transaction_price)
                if self.asks[0].who == 201:
                    lob_agent.b.trader_info[self.asks[0].who]['money'] += transaction_price    
                    

                '''
                print('**matched** bid price: ', self.bids[0].price, self.bids[0].who,
                      ', ask price: ', self.asks[0].price, self.asks[0].who,
                      ', match: ', transaction_price)
                '''
                
                self.stock_price = transaction_price
                self.count += 1
                
                # Determine the transaction quantity as the minimum of the quantities of the best bid and ask orders
                transaction_quantity = min(self.bids[0].quantity, self.asks[0].quantity)

                
                # Update the quantities of the best bid and ask orders
                self.bids[0].quantity -= transaction_quantity
                self.asks[0].quantity -= transaction_quantity
                
                # Remove any orders that have been completely filled
                if self.bids[0].quantity == 0:
                    self.bids.pop(0)
                if self.asks[0].quantity == 0:
                    self.asks.pop(0)
                
                tranList.append(transaction_price)
               
                

            else:
                if tranList != []:
                    close = statistics.median(tranList)
                    self.Close.append(close)
                
                # If the best bid price is lower than the best ask price, there is no match
                break
        
        # print("stepid=",self.stepid)
        self.stepid += 1
        
    def print_order_book(self):
        """
        Print out the current state of the order book along with the stock price.
        """
        print("Bids:")
        for order in self.bids:
            print(order)

        print("Asks:")
        for order in self.asks:
            print(order)

        print("Stock Price:", self.stock_price)
        

class Order:
    """
    Represent an order in the limit order book
    """
    def __init__(self, order_type, quantity, price, who, stepid):
        self.type = order_type # a buy order (type='buy') or a sell order (type='sell')
        self.quantity = quantity
        self.price = price
        self.who = who
        self.stepid = stepid
    def __str__(self):
        """
        Return a string representation of the Order object 
        """
        return f"{self.type} {self.quantity}@{self.price} {self.who} {self.stepid}"
    

class Chartists:
    def __init__(self, num_traders=20):
        self.lob = LimitOrderBook()
        self.num_traders = num_traders
        self.close = 0
        self.asks = 0
        self.bids = 0
        self.macd_signal = []
        self.trader_info = {}
        for i in range(181, (181+num_traders)):
            self.trader_info[i] = {
                'shares': 80,
                'money': 1000
            }
        ##self.strategy_duration = 150 #原本的paper有提到這個，如果在150 ticks中策略沒有被執行的話就停，但如果是取市價直接成交的話應該就不需要了吧（？）


    def run(self, fast_period=12, slow_period=26, signal_period=9):
        close_price = np.array(lob_agent.StockPrices)
        ## print("self.close",close_price)
        # Calculate the short-term EMA and long-term EMA
        ema_fast = pd.Series(close_price).ewm(span=fast_period, min_periods=fast_period).mean()
        ema_slow = pd.Series(close_price).ewm(span=slow_period, min_periods=slow_period).mean()
        
         # Calcualte the MACD line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()

        # Compute the MACD signal as a crossover of the MACD and signal lines
        self.macd_signal = np.where(macd > signal, 1, 0)


class BigInvestor:
    def __init__(self, buy_ratio=0.75, sell_ratio=0.2, mutiplier = 100000):
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.strategy_duration = 100
        self.strategy_tick = 1
        self.c = Chartists()
        self.lob = LimitOrderBook()
        self.close = 0
        self.asks = 0
        self.bids = 0
        self.la = 0
        self.lb = 0
        self.macd_signal = []
        self.trader_info = {}
        self.asset = []
        for j in range(201, 202):
            self.trader_info[j] = {
                'shares': 80 * mutiplier,
                'money': 1000 * mutiplier
        }
        

    def run(self, fast_period=12, slow_period=26, signal_period=9):
        close_price = np.array(lob_agent.StockPrices)
        ##print("self.close",self.close)
        # Calculate the short-term EMA and long-term EMA
        ema_fast = pd.Series(close_price).ewm(span=fast_period, min_periods=fast_period).mean()
        ema_slow = pd.Series(close_price).ewm(span=slow_period, min_periods=slow_period).mean()
        
         # Calcualte the MACD line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()

        # Compute the MACD signal as a crossover of the MACD and signal lines
        self.macd_signal = np.where(macd > signal, 1, 0)
        # print(self.trader_info[201]['shares'])
        ##print('asset=',self.trader_info[201]['shares']*close_price[-1]+self.trader_info[201]['money'])
        self.asset.append(self.trader_info[201]['shares']*close_price[-1]+self.trader_info[201]['money'])

    
    
    
class RandomTrader:
    def __init__(self, initial_price, time_steps, num_RandomTraders, big_strategy = True, c_strategy=True, sellfirst_strategy=True, b_buy_ratio=0.75, b_sell_ratio=0.2, b_multiplier = 100000):
        self.b_buy_ratio = b_buy_ratio
        self.b_sell_ratio = b_sell_ratio
        self.sellfirst_strategy = sellfirst_strategy
        self.c_strategy = c_strategy
        self.big_strategy = big_strategy
        self.P0 = initial_price
        self.T = time_steps
        self.lob = LimitOrderBook()
        self.c = Chartists()
        self.b = BigInvestor(buy_ratio=b_buy_ratio, sell_ratio=b_sell_ratio, mutiplier = b_multiplier)
        self.lob.transactions = []
        self.sigma = 0.02 / np.sqrt(252)
        self.StockPrices = []
        self.num_traders = num_RandomTraders
        self.stepid = 0
        self.flag1 = 2
        self.flag2 = 2
        self.buyfirst = 1
        self.sellfirst = 1
        self.trader_info = {}
        self.lala = []
        self.BGtimedivided = [0]
        self.Ltimedivided = [0]
        self.bgcolor = ['#FFFFFF']
        for i in range(1, num_RandomTraders+1):
            self.trader_info[i] = {
                'shares': 80,
                'money': 1000
            }

        self.sigmaList_stat = []
        self.r_wealth = []
        self.b_wealth = []
        self.c_wealth = []
        self.LogReturn = [0]
        self.Volume = []

    def run(self):

        for t in range(self.T):
            self.stepid = t
            if t > 0:
                prices = [transaction for transaction in self.lob.Close[-20:]]
                if len(prices) < 2:
                    self.sigma = 0.02 / np.sqrt(252)
                else:
                    self.sigma = np.std(prices) / np.mean(prices)
                    # self.sigma = statistics.stdev(prices)/statistics.mean(prices)

            mu_b = 1.01
            mu_s = 0.99

            # Each trader can buy, sell, or hold with probability based on their trend
            actions = ['buy', 'sell', 'hold']
            trends = []
            for i in range(1, self.num_traders+1):
                if self.trader_info[i]['shares'] == 0:
                    trends.append([1/4, 0, 3/4])
                elif self.trader_info[i]['money'] == 0:
                    trends.append([0, 1/4, 3/4])
                else:
                    trends.append([1/3, 1/3, 1/3])

            for i in range(1, self.num_traders+1):
                action = random.choices(actions, weights=trends[i-1])[0]
            
                if action == 'buy':
                    price_b = self.P0 * random.gauss(mu_b, self.sigma)
                    if self.trader_info[i]['money'] >= price_b:
                        self.lob.insert_order(Order('buy', 1, price_b, i, self.stepid))
                        self.trader_info[i]['money'] -= price_b
                        
                elif action == 'sell':
                    if self.trader_info[i]['shares'] > 0:
                        price_s = self.P0 * random.gauss(mu_s, self.sigma)
                        self.lob.insert_order(Order('sell', 1, price_s, i, self.stepid))
                        self.trader_info[i]['shares'] -= 1
            
            if t >= 26:
                if self.c_strategy == True:
                    self.c.run()
                    for i in range(self.num_traders+1, (self.num_traders+1 + self.c.num_traders)):
                        price_b = self.c.asks
                        if  self.c.macd_signal[-1] == 1 and self.c.macd_signal[-2] == 0 and self.c.trader_info[i]['money'] >= price_b:
                            self.lob.insert_order(Order('buy', 1, price_b, i, self.stepid))
                            self.c.trader_info[i]['money'] -= price_b
                            
                        elif self.c.macd_signal[-1] == 0 and self.c.macd_signal[-2] == 1 and self.c.trader_info[i]['shares'] > 0:
                            price_s = self.c.bids
                            self.lob.insert_order(Order('sell', 1, price_s, i, self.stepid))
                            self.c.trader_info[i]['shares'] -= 1
                    
                    
                if self.big_strategy == True:
                    self.b.run()
                    if (self.b.macd_signal[-1] == 1 and self.b.macd_signal[-2] == 0 and self.flag2 == 2) or self.flag1 == 1:
                        self.flag1 = 1
                        self.lala.append(t)
                        initial_price = self.lob.Close[-1]
                        price_b = self.b.asks
                        if self.buyfirst <= self.b.strategy_duration * 0.5:
                            self.BGtimedivided.append(t)
                            self.bgcolor.append('#20B2AA')
                            quantity_b = self.b.buy_ratio * self.b.la # 收購數量是「收購比率」*「目前LOB還沒match的賣單數量」
                            # print('qb=',quantity_b,self.b.buy_ratio ,self.b.la)
                            for i in range(int(quantity_b)):
                                if self.b.trader_info[self.num_traders+self.c.num_traders+1]['money'] >= price_b: 
                                    self.lob.insert_order(Order('buy', 1, price_b, self.num_traders+self.c.num_traders+1, self.stepid))

                                ##else: # 如果收購到一半沒有錢了就停
                                    ##pass
                            self.b.strategy_tick += 1
                            self.buyfirst += 1
                        

                        elif self.b.strategy_duration * 0.5 < self.buyfirst and self.buyfirst <= self.b.strategy_duration * 0.75 :
                            self.BGtimedivided.append(t)
                            self.bgcolor.append('#FFDEAD')
                            self.b.strategy_tick += 1
                            self.buyfirst += 1

                        else:
                            if self.buyfirst < self.b.strategy_duration:
                                self.BGtimedivided.append(t)
                                self.bgcolor.append('#FF7F50')
                                self.b.strategy_tick += 1
                                self.buyfirst += 1
                                if self.b.lb != 0 and self.b.close[-1] >= initial_price:
                                    quantity_s = self.b.sell_ratio * self.b.lb
                                    # print('qs=',quantity_s, self.b.sell_ratio, self.b.lb)
                                    for i in range(int(quantity_s)):
                                        if self.b.trader_info[self.num_traders+self.c.num_traders+1]['shares'] > 0:
                                            price_s = self.b.bids
                                            self.lob.insert_order(Order('sell', 1, price_s, self.num_traders+self.c.num_traders+1, self.stepid))
                                            self.b.trader_info[self.num_traders+self.c.num_traders+1]['shares'] -= 1
                                        else:
                                            self.BGtimedivided.append(t)
                                            self.bgcolor.append('#FF7F50')
                                            pass
                            '''            
                            if self.buyfirst < self.b.strategy_duration:
                                self.b.strategy_tick += 1
                                self.buyfirst += 1'''
                                
                        
                        
                        if self.buyfirst == 100:
                            self.flag1 = 2
                            self.buyfirst = 0
                            self.lala.append("a")
                            self.lala.append(t)
                            self.BGtimedivided.append(t)
                            self.bgcolor.append('#FF7F50')

                    elif ((self.b.macd_signal[-1] == 0 and self.b.macd_signal[-2] == 1 and self.flag1 == 2) or self.flag2 == 1) and self.sellfirst_strategy:
                        self.flag2 = 1
                        initial_price = self.lob.Close[-1]
                        if self.sellfirst <= self.b.strategy_duration * 0.5:
                            quantity_s = self.b.buy_ratio * self.b.lb #改這裡
                            # print('qs=',quantity_s,self.b.buy_ratio, self.b.lb) #改這裡
                            for i in range(int(quantity_s)):
                                if self.b.trader_info[self.num_traders+self.c.num_traders+1]['shares'] > 0:
                                    price_s = self.b.bids
                                    self.lob.insert_order(Order('sell', 1, price_s, self.num_traders+self.c.num_traders+1, self.stepid))
                                    self.b.trader_info[self.num_traders+self.c.num_traders+1]['shares'] -= 1

                                ##else: # 如果收購到一半沒有錢了就停
                                    ##pass

                            self.b.strategy_tick += 1
                            self.sellfirst += 1

                        elif self.b.strategy_duration * 0.5 < self.sellfirst and self.sellfirst <= self.b.strategy_duration * 0.75 :
                            self.b.strategy_tick += 1
                            self.sellfirst += 1

                        else:
                            if self.b.la != 0 and self.b.close[-1] <= initial_price:
                                quantity_b = self.b.sell_ratio * self.b.la #改這裡
                                # print('qb=',quantity_b, self.b.sell_ratio, self.b.la) #改這裡
                                price_b = self.b.asks
                                for i in range(int(quantity_b)):
                                    if self.b.trader_info[self.num_traders+self.c.num_traders+1]['money'] >= price_b: 
                                        self.lob.insert_order(Order('buy', 1, price_b, self.num_traders+self.c.num_traders+1, self.stepid))
        
                                    else: # 如果收購到一半沒有錢了就停
                                        pass
        
                                self.b.strategy_tick += 1
                                self.sellfirst += 1
                        
                        if self.sellfirst == 100:     
                            self.flag2 = 2
                            self.sellfirst = 0
                            self.lala.append("b")
                            self.lala.append(t)

                            self.BGtimedivided.append(t)
                            self.bgcolor.append('#ebebeb')

                            if t+100 < self.T:
                                self.BGtimedivided.append(t+100)
                                self.bgcolor.append('#FFFFFF')
                    

                    else:
                        self.BGtimedivided.append(t)
                        self.bgcolor.append('#FFFFFF')
            # print(f"Time step {t}:")
            # print(self.c.macd_signal)

            self.lob.match_orders()
            
            self.P0 = self.lob.stock_price
            self.c.close = self.lob.Close
            
            if self.lob.asks == []:
                self.c.asks = self.lob.Close[-1]
                self.b.asks = self.lob.Close[-1]
            else:
                self.c.asks = self.lob.asks[0].price
                self.b.asks = self.lob.asks[int(len(self.lob.asks)*0.4)].price
                
            if self.lob.bids == []:
                self.c.bids = self.lob.Close[-1]
                self.b.bids = self.lob.Close[-1]
            else:
                self.c.bids = self.lob.bids[0].price
                self.b.bids = self.lob.bids[int(len(self.lob.bids)*0.15)].price
                
            self.b.close = self.lob.Close
            
            '''
            self.b.asks = self.lob.asks[int(len(self.lob.asks)*0.1)].price
            self.b.bids = self.lob.bids[int(len(self.lob.bids)*0.02)].price
            '''


            self.b.la = len(self.lob.asks)
            ##print("la",self.b.la)
            self.b.lb = len(self.lob.bids)
            ##print("lb",self.b.lb)
            


            # Update Dataframe
            self.StockPrices.append(self.P0) # Stock Prices

            # Update sigma
            prices = [transaction for transaction in self.StockPrices[-20:]]
            if len(prices) < 2:
                self.sigma = 0.02 / np.sqrt(252)
            else:
                self.sigma = np.std(prices) / np.mean(prices)

            self.sigmaList_stat.append(self.sigma) # Sigma
            

            if len(self.StockPrices) >= 2:
                self.LogReturn.append(np.log(self.P0)-np.log(self.StockPrices[-2])) # Vol

            ttl_r_shares = 0
            ttl_r_money = 0
            for i in range(1, self.num_traders+1):
                ttl_r_shares +=  self.trader_info[i]['shares']
                ttl_r_money += self.trader_info[i]['money']
            self.r_wealth.append(ttl_r_shares*self.P0 + ttl_r_money)
            
            
            ttl_c_shares = 0
            ttl_c_money = 0
            for i in range(self.num_traders+1, self.num_traders+1+self.c.num_traders):
                ttl_c_shares +=  self.c.trader_info[i]['shares']
                ttl_c_money += self.c.trader_info[i]['money']
            self.c_wealth.append(ttl_c_shares*self.P0 + ttl_c_money)

            self.b_wealth.append(self.b.trader_info[self.num_traders+self.c.num_traders+1]['shares']*self.P0+ self.b.trader_info[self.num_traders+self.c.num_traders+1]['money'])
            

            
            self.Volume.append(self.lob.count)
            # print('number of matched orders:', self.lob.count)
            # print(self.trader_info)
            # print('Limit Order Book-------------------------', self.P0)
            # self.lob.print_order_book()
            # print()

        self.df = pd.DataFrame({
            "Close": self.StockPrices,
            "Volume": self.Volume,
            "Sigma": np.array(self.sigmaList_stat) * np.sqrt(252),
            "Variance": (np.array(self.sigmaList_stat) * np.sqrt(252))**2,
            "LogReturn": np.array(self.LogReturn),
            "Random Agents": self.r_wealth,
            "Chartists": self.c_wealth,
            "Big Investor": self.b_wealth
        })
        self.BGtimedivided.append(self.T)
        self.bgcolor.append('#FFFFFF')

        
                        
    
"""
參數在這裡調整
@ 初始股價
@ 模擬時長 (500-1000)
@ Random Traders數 (寫死了，不要動比較好)
@ 有/沒有 加入big investor
@ 有/沒有 加入chartists
@ 有/沒有 加入sell first
@ 財團買入佔賣單比例 (75-95%)
@ 財團拋售佔買單比例 (20-35%)
@ 財團初始資產槓桿
"""

'''random traders only'''
lob_agent = RandomTrader(initial_price=10, time_steps=2000, num_RandomTraders=180, 
                         big_strategy=False, c_strategy=False, sellfirst_strategy=False,
                         b_buy_ratio=0.75, b_sell_ratio=0.2, b_multiplier = 100000)      ##(step0 stock price, ticks, number of random agent)


"""
這裡是畫股價變化圖
"""
lob_agent.run()
#print(lob_agent.lala)           
#print(lob_agent.BGtimedivided) 
#print(lob_agent.bgcolor)
            
# Plot the stock price over time
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("Stock Price")
plt.title("Stock Price Over Time (Only RA)")
plt.plot(lob_agent.StockPrices)
time = lob_agent.BGtimedivided
color = lob_agent.bgcolor
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

"""
sigma 變化圖
"""
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("Sigma")
plt.title("Sigma Over Time (Only RA)")
plt.plot(lob_agent.df["Sigma"])
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

df1 = lob_agent.df
sigma_1 = lob_agent.df['Sigma']



''' no big invester '''
lob_agent = RandomTrader(initial_price=10, time_steps=2000, num_RandomTraders=180, 
                         big_strategy=False, c_strategy=True, sellfirst_strategy=False,
                         b_buy_ratio=0.75, b_sell_ratio=0.2, b_multiplier = 100000)      ##(step0 stock price, ticks, number of random agent)



"""
這裡是畫股價變化圖
"""
lob_agent.run()
#print(lob_agent.lala)           
#print(lob_agent.BGtimedivided) 
#print(lob_agent.bgcolor)
            
# Plot the stock price over time
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("Stock Price")
plt.title("Stock Price Over Time (RA + Chartists)")
plt.plot(lob_agent.StockPrices)
time = lob_agent.BGtimedivided
color = lob_agent.bgcolor
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

"""
sigma 變化圖
"""
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("Sigma")
plt.title("Sigma Over Time  (RA + Chartists)")
plt.plot(lob_agent.df["Sigma"])
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

df2= lob_agent.df
sigma_2 = lob_agent.df['Sigma']






lob_agent = RandomTrader(initial_price=10, time_steps=2000, num_RandomTraders=180, 
                         big_strategy=True, c_strategy=True, sellfirst_strategy=False,
                         b_buy_ratio=0.75, b_sell_ratio=0.2, b_multiplier = 100000)      ##(step0 stock price, ticks, number of random agent)


"""
這裡是畫股價變化圖
"""
lob_agent.run()
#print(lob_agent.lala)           
#print(lob_agent.BGtimedivided) 
#print(lob_agent.bgcolor)
            
# Plot the stock price over time
plt.figure(figsize=(15,8))
# plt.xlabel("Time Step");plt.ylabel("Stock Price")
plt.title("Stock Price Over Time (With the Big Investor)")
plt.plot(lob_agent.StockPrices)
time = lob_agent.BGtimedivided
color = lob_agent.bgcolor
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

"""
sigma 變化圖
"""
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("Sigma")
plt.title("Sigma Over Time")
plt.plot(lob_agent.df["Sigma"])
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

df3 = lob_agent.df
sigma_3 = lob_agent.df['Sigma']

"""
這裡是畫財團資產變動圖
"""
# Plot the bi asset over time
plt.figure(figsize=(10,8))
# plt.xlabel("Time Step");plt.ylabel("asset")
plt.title("Asset of the Big Investor Over Time (With the Big Investor)")
plt.plot(lob_agent.df["Big Investor"])
for idx in range(len(time)-1):
    plt.axvspan(time[idx], time[idx+1], facecolor=color[idx], alpha=0.5)
plt.show()

"""
這裡有一些可能用得到的東西，可以試
    (1)只有random agents (2) random agents+chartists (3) random agents+chartists+big investor 
    這三種狀況的差別
"""
# plt.plot(np.)
# plt.plot(np.log(df["Sigma"]))
# plt.plot(np.log(df["volatility"]))
# plt.hist(Volume)

"""
這裡是印出dataframe
"""
print("Simulation data")
df=lob_agent.df
print(df)




"""
sigma
"""
plt.plot(sigma_1, c='black')
plt.plot(sigma_2, c='blue')
plt.plot(sigma_3, c='red')
# plt.xlabel("time");plt.ylabel=("sigma")
plt.title("Simga Comparison")
plt.show()


"""
sigma hist
"""

fig, (ax3, ax2, ax1) = plt.subplots(3, sharex=True)
fig.suptitle('Simga Histogram')
ax3.hist(sigma_3, color='red')
ax2.hist(sigma_2, color='blue')
ax1.hist(sigma_1, color='black')
plt.xlabel("Simga")
plt.ylabel=("Freq")
plt.show()

"""
Return 
"""
return_df = pd.DataFrame({
    "Random Agents":[(df["Random Agents"][len(df["Random Agents"])-1]-df["Random Agents"][0])/df["Random Agents"][0]],
    "Chartists":[(df["Chartists"][len(df["Chartists"])-1]-df["Chartists"][0])/df["Chartists"][0]],
    "Big Investor":[(df["Big Investor"][len(df["Big Investor"])-1]-df["Big Investor"][0])/df["Big Investor"][0]],
},index=["Return rate"])
print(return_df)

