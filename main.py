import requests
import json

key = ""
with open("tiingoKey.txt", "r") as data:
    key = data.read().strip()

headers = {
            'Content-Type': 'application/json'
}
requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/aapl/prices?startDate=2014-11-21&endDate=2024-11-21&format=json&resampleFreq=daily&token={key}", headers=headers)
with open("oneDayOfData.json", "w") as file:
    json.dump(requestResponse.json(), file)

