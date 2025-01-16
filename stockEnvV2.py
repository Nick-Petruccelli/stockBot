import datetime
import json
import numpy as np

class StocksEnv():
    def __init__(self, init_balance=10000, episode_len=60, stocks_per_step=10, past_days_looked_at=7):
        self.train_stocks = ["appl","msft","amzn","nvda","googl","tsla","goog","brk.b","meta","unh",
                        "xom","lly","jpm","jnj","v","pg","ma","avgo","hd","cvx"]
        self.init_balance = init_balance
        self.amount_per_step = stocks_per_step
        self.stock_index = 0
        self.time_period_index = past_days_looked_at
        self.episode_len = episode_len
        self.past_days_looked_at = past_days_looked_at
        self.state = None
        self.in_episode = False

    def start_next_episode(self):
        self.cur_stock = self.train_stocks[self.stock_index]
        self.balance = self.init_balance
        self.stocks_held = 0
        self.state_index = self.past_days_looked_at
        self.data = self.get_data(self.cur_stock, self.time_period_index)
        while self.data == []:
            self.stock_index += 1
            self.time_period_index = self.past_days_looked_at
            if self.stock_index >= len(self.train_stocks):
                return np.array([])
            self.cur_stock = self.train_stocks[self.stock_index]
            self.data = self.get_data(self.cur_stock, self.time_period_index)
        self.time_period_index += self.episode_len
        self.in_episode = True
        self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
        return self.state.to_array()

    def get_data(self, stock, time_period_idx):
        with open(f'data/{stock}Data.json', 'r') as file:
            data = json.load(file)
            if self.time_period_index+self.episode_len >= len(data):
                return []
            return data[time_period_idx-self.past_days_looked_at:time_period_idx+self.episode_len]

    def step(self, action):
        #action is int 0:=buy 1:=sell
        if not self.in_episode:
            raise Exception("Enviorment cant step if not in episode")
        self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
        if action == 0:
            if self.balance <= 0:
                self.state_index += 1
                done = False
                if self.state_index >= len(self.data)-1:
                    done = True
                    self.in_episode = False
                else:
                    self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
                return self.state.to_array(), 0, done, None
            self.stocks_held += self.amount_per_step
            self.balance -= self.amount_per_step * self.state.open_prices[0]
            reward = 0
            self.state_index += 1
            done = False
            if self.state_index == len(self.data)-1:
                done = True
                self.in_episode = False
            else:
                self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
            return self.state.to_array(), reward, done, None
        if action == 1:
            if self.stocks_held <= 0:
                self.state_index += 1
                done = False
                if self.state_index >= len(self.data)-1:
                    done = True
                    self.in_episode = False
                else:
                    self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
                return self.state.to_array(), 0, done, None
            self.stocks_held -= self.amount_per_step
            reward = self.amount_per_step * self.state.open_prices[0]
            self.balance += reward
            self.state_index += 1
            done = False
            if self.state_index == len(self.data)-1:
                done = True
                self.in_episode = False
            else:
                self.state = State(self.data, self.state_index, self.past_days_looked_at, self.balance, self.stocks_held)
            return self.state.to_array(), reward, done, None
        return np.array([]), None, None, None


class State():
    def __init__(self, data, state_index, past_days_looked_at, balance, stock_held):
        data_range = data[state_index-past_days_looked_at: state_index]
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

    def to_array(self):
        out = []
        for close in self.close_prices:
            out.append(close)
        for open in self.open_prices:
            out.append(open)
        for vol in self.volume:
            out.append(vol)
        out.append(self.day)
        out.append(self.month)
        out.append(self.balance)
        out.append(self.stock_held)
        return np.array(out)


if __name__ == "__main__":
    env = StocksEnv()
    training = False
    ep_count = 0
    while not training:
        out = env.start_next_episode()
        ep_count += 1
        if out == None:
            break
        in_ep = True
        count = 0
        while(in_ep):
            state, reward, done, _ = env.step(count % 2)
            #print("step: ", count, " reward: ", reward)
            count += 1
            in_ep = not done
    print("epCount: ", ep_count)
