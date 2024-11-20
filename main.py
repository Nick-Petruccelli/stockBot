import requests

key = ""
with open("tiingoKey.txt", "r") as data:
    key = data.readline()
print(key)
