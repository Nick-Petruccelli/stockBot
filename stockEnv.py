import requests
import json

class StocksEnv():
    def __init__(self, init_balance=10000, train_block=28):
        self.train_stocks = ["aapl","msft","amzn","nvda","googl","tsla","goog","brk.b","meta","unh",
                        "xom","lly","jpm","jnj","v","pg","ma","avgo","hd","cvx"]
        self.init_balance = init_balance
        self.train_block = train_block
        self.stock_index = 0
        self.start_date = "2014-11-21"

    def start_episode(self, stock, start_date, end_date):
        self.date = start_date
        self.balance = self.init_balance
        self.data = 
        
    def step(action):
        #action is int 0:=buy 1:=sell 2:=hold
        
