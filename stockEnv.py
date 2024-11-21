import datetime

class StocksEnv():
    def __init__(self, init_balance=10000, train_block=28, stocks_per_step=10, past_days_looked_at=7):
        self.train_stocks = ["aapl","msft","amzn","nvda","googl","tsla","goog","brk.b","meta","unh",
                        "xom","lly","jpm","jnj","v","pg","ma","avgo","hd","cvx"]
        self.init_balance = init_balance
        self.amount_per_step = stocks_per_step
        self.train_block = train_block
        self.stock_index = 0
        self.past_days_looked_at = past_days_looked_at

    def start_episode(self, stock, start_date, end_date):
        self.balance = self.init_balance
        self.stocks_held = 0
        self.state_index = 0
        self.data = self.getData(self.stock_index)
        self.in_episode = True
        
    def step(self, action):
        #action is int 0:=buy 1:=sell 2:=hold
        if not self.in_episode:
            raise Exception("Enviorment cant step if not in episode")
        if action == 0:
            self.stocks_held += self.amount_per_step * self.state.open
            self.balance -= self.amount_per_step
            reward = 0
            self.state_index += 1
            return 


class State():
    def __init__(self, data, state_index, past_days_looked_at, balance, stock_held):
        data_range = [data[i] for i in range(state_index, state_index-past_days_looked_at, -1)]
        self.close_prices = [obs["close"] for obs in data_range]
        self.open_prices = [obs["open"] for obs in data_range]
        self.volume = [obs["volume"] for obs in data_range]
        date_str = data[state_index]["date"]
        date_str = date_str[:10]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        self.day = date.weekday()
        self.month = date.month
        self.balance = balance
        self.stock_held = stock_held

