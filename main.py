import requests

key = ""
with open("tiingoKey.txt", "r") as data:
    key = data.read().strip()

headers = {
            'Content-Type': 'application/json'
}
requestResponse = requests.get(f"https://api.tiingo.com/tiingo/crypto/prices?tickers=btcusd&startDate=2019-01-02&resampleFreq=5min&token={key}", headers=headers)
print(requestResponse.json())
