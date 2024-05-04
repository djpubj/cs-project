from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__, static_url_path='/static')

def crpto(a):
    if a == 1:
        return "BTC-USD"
    elif a == 2:
        return "ETH-USD"
    elif a == 3:
        return "DOGE-USD"
    elif a == 4:
        return "STETH-USD"
    elif a == 5:
        return "USDT-USD"

def generate_random_scenarios(num_scenarios):
    volatility_mean = 0.1
    volatility_std = 0.02
    trading_volume_mean = 1e8
    trading_volume_std = 1e7

    volatility_scenarios = np.random.normal(volatility_mean, volatility_std, num_scenarios)
    trading_volume_scenarios = np.random.normal(trading_volume_mean, trading_volume_std, num_scenarios)

    return volatility_scenarios, trading_volume_scenarios

def monte_carlo_simulation_for_next_day(x, num_simulations, volatility_std, trading_volume_std, reg):
    predicted_open_prices = []
    predicted_close_prices = []

    for i in range(num_simulations):
        modified_x = x.copy()
        modified_volatility = np.random.normal(0, volatility_std)
        modified_trading_volume = np.random.normal(0, trading_volume_std)
        modified_x["prev_close"] += modified_volatility
        modified_x["prev_open"] += modified_trading_volume

        next_day_prediction = reg.predict([[modified_x["prev_close"].iloc[-1], modified_x["prev_open"].iloc[-1]]])

        predicted_open_prices.append(next_day_prediction[0][0])
        predicted_close_prices.append(next_day_prediction[0][1])

    return predicted_open_prices, predicted_close_prices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    p = int(request.form['crypto'])
    ticker = crpto(p)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3 * 365)

    data = yf.download(ticker, start=start_date, end=end_date)

    csv_file_path = "btc_dt.csv"
    data.to_csv(csv_file_path, index=True)

    df = pd.read_csv("btc_dt.csv")
    series_shifted_close = df["Close"].shift()
    series_shifted_open = df["Open"].shift()
    df["prev_close"] = series_shifted_close
    df["prev_open"] = series_shifted_open

    df.dropna(inplace=True)

    x = df[["prev_close", "prev_open"]]
    y = df[["Open", "Close"]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    predicted = reg.predict(x_test)

    acc = reg.score(x_test, y_test)

    mae_open = metrics.mean_absolute_error(y_test["Open"].values, predicted[:, 0])
    mse_open = metrics.mean_squared_error(y_test["Open"].values, predicted[:, 0])
    rmse_open = np.sqrt(metrics.mean_squared_error(y_test["Open"].values, predicted[:, 0]))

    mae_close = metrics.mean_absolute_error(y_test["Close"].values, predicted[:, 1])
    mse_close = metrics.mean_squared_error(y_test["Close"].values, predicted[:, 1])
    rmse_close = np.sqrt(metrics.mean_squared_error(y_test["Close"].values, predicted[:, 1]))

    accuracy_open = pd.DataFrame({
        "mae_open": [mae_open],
        "mse_open": [mse_open],
        "rmse_open": [rmse_open]
    })

    accuracy_close = pd.DataFrame({
        "mae_close": [mae_close],
        "mse_close": [mse_close],
        "rmse_close": [rmse_close]
    })

    next_day_prediction = reg.predict([[df["prev_close"].iloc[-1], df["prev_open"].iloc[-1]]])

    last_date_row = df.iloc[-1]
    last_date_str = last_date_row["Date"]
    last_date = pd.to_datetime(last_date_str)

    next_day_date = last_date + pd.Timedelta(days=1)

    plot_data = pd.DataFrame({
        "Date": df.iloc[x_test.index - 1]["Date"].values,
        "Actual_Open": y_test["Open"].values,
        "Actual_Close": y_test["Close"].values,
        "Predicted_Open": predicted[:, 0],
        "Predicted_Close": predicted[:, 1]
    })

    plt.figure(figsize=(10, 6))
    plt.plot(plot_data["Date"], plot_data["Actual_Open"], label="Actual Open Price", marker="o", color='blue')
    plt.plot(plot_data["Date"], plot_data["Actual_Close"], label="Actual Close Price", marker="o", color='green')
    plt.plot(plot_data["Date"], plot_data["Predicted_Open"], label="Predicted Open Price", marker="o", color='red')
    plt.plot(plot_data["Date"], plot_data["Predicted_Close"], label="Predicted Close Price", marker="o", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices with Dates")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/plot.png')

    differences = y_test["Open"].values - predicted[:, 0]
    plt.figure(figsize=(10, 6))
    plt.hist(differences, bins=30, edgecolor='black')
    plt.xlabel("Difference (Actual_open - Predicted_open)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Differences between Actual and Predicted Values")
    plt.savefig('static/plot_diff_open.png')

    difference = y_test["Close"].values - predicted[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(difference, bins=30, edgecolor='black')
    plt.xlabel("Difference (Actual_close- Predicted_close)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Differences between Actual and Predicted Values")
    plt.savefig('static/plot_diff_close.png')

    volatility_std = df["Close"].std()
    trading_volume_std = df["Open"].std()

    num_simulations = 100
    monte_carlo_open_prices, monte_carlo_close_prices = monte_carlo_simulation_for_next_day(df.tail(1), num_simulations, volatility_std, trading_volume_std, reg)

    mean_open_price = np.mean(monte_carlo_open_prices)
    std_open_price = np.std(monte_carlo_open_prices)
    mean_close_price = np.mean(monte_carlo_close_prices)
    std_close_price = np.std(monte_carlo_close_prices)

    return render_template('results.html', accuracy_score=acc,next_day_date=next_day_date, next_day_prediction=next_day_prediction,table3=tabulate(accuracy_open, headers='keys', tablefmt='html'), table4=tabulate(accuracy_close, headers='keys', tablefmt='html'), mean_open_price=mean_open_price, std_open_price=std_open_price, mean_close_price=mean_close_price, std_close_price=std_close_price,plot='plot.png', plot_diff_open='plot_diff_open.png', plot_diff_close='plot_diff_close.png')

if __name__ == '__main__':
    app.run(debug=True)
