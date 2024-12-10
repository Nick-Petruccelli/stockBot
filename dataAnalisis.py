import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

stocks = ["appl","msft","amzn","nvda","googl","tsla","goog","brk.b","meta","unh",
                "xom","lly","jpm","jnj","v","pg","ma","avgo","hd","cvx"]

for stock in stocks:
    with open(f'data/{stock}Data.json', 'r') as file:
        data = json.load(file)
    open_prices = [e['open'] for e in data if type(e) == dict]
    close_prices = [e['close'] for e in data if type(e) == dict]
    volume = [e['volume'] for e in data if type(e) == dict]
    fig, ax = plt.subplots()
    plt.plot(open_prices, label="open")
    plt.plot(close_prices, label="close")
    ax.set_xlabel('Time')
    ax.set_ylabel('Open-Close')
    ax.set_title(f'{stock} Open Close Vol')
    ax.legend()
    plt.savefig(f"plots/{stock}OpenClose.png")
    plt.clf()
    fig, ax = plt.subplots()
    plt.plot(volume, label="volume")
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume')
    ax.set_title(f'{stock} Open Close Vol')
    ax.legend()
    plt.savefig(f"plots/{stock}Vol.png")
    plt.clf()
    df = pd.DataFrame({"open": open_prices, "close": close_prices, "volume": volume})
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f"plots/{stock}CorrMat.png")
    plt.clf()
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=len(df.columns), figsize=(10, 10))
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i == j:
                axes[i, j].hist(df[col1])  # Diagonal: histograms
            else:
                axes[i, j].scatter(df[col1], df[col2])  # Off-diagonal: scatter plots
    for i, col in enumerate(df.columns):
        axes[i, 0].set_ylabel(col)
        axes[-1, i].set_xlabel(col)
    plt.tight_layout()
    plt.savefig(f"plots/{stock}PairPlot.png")
    plt.clf()
