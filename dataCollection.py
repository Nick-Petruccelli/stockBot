import requests
import json

key = ""
with open("tiingoKey.txt", "r") as data:
    key = data.read().strip()

headers = {
            'Content-Type': 'application/json'
}
train_stocks = ["aapl","msft","amzn","nvda","googl","tsla","goog","brk.b","meta","unh",
                "xom","lly","jpm","jnj","v","pg","ma","avgo","hd","cvx"]

for ticker in train_stocks:
    requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=2014-11-21&endDate=2024-11-21&format=json&resampleFreq=daily&token={key}", headers=headers)
    with open(f"data/{ticker}Data.json", "w") as file:
        json.dump(requestResponse.json(), file)
