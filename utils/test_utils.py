import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored

def parse_dates(df):
    return df['Date'].head(1).values[0].split(' ')[0], df['Date'].tail(1).values[0].split(' ')[0]

def plot_graph_price_diff(test_df, LOOKUP_STEP=0):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_Close_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'Close_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def plot_graph_price(df):
    # take a merged csv and split here
    # print(df.tail())
    dates = []
    BTC_data = df[df['Symbol'] == 'BTC']
    ETH_data = df[df['Symbol'] == 'ETH']
    ADA_data = df[df['Symbol'] == 'ADA']
    dates.append(parse_dates(BTC_data))
    dates.append(parse_dates(ETH_data))
    dates.append(parse_dates(ADA_data))
    print('start and closing dates', *dates, sep='\n')
    plt.plot(BTC_data['Date'], BTC_data['Close'], c='b')
    plt.plot(ETH_data['Date'], ETH_data['Close'], c='r')
    plt.plot(ADA_data['Date'], ADA_data['Close'], c='g')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(["Bitcoin", "Ethereum", "Cardano"])
    plt.show()

def plot_graph_run_stats(df):
    df = df.assign(MAE_Percentage = lambda x: (x['MAE']/x['LatestPrice'])*100) # divide each val by latest price
    BTC_data = df[df['TKR'] == 'BTC'].reset_index(drop=True)
    ETH_data = df[df['TKR'] == 'ETH'].reset_index(drop=True)
    ADA_data = df[df['TKR'] == 'ADA'].reset_index(drop=True)
    plt.plot(BTC_data['MAE_Percentage'] , c='b')
    plt.plot(ETH_data['MAE_Percentage'] , c='r')
    plt.plot(ADA_data['MAE_Percentage'] , c='g')
    plt.xlabel("Runs")
    plt.ylabel("Relative Mean Abs. Error (%)")
    plt.legend(["Bitcoin", "Ethereum", "Cardano"])
    plt.show()

def get_final_df(model, data, LOOKUP_STEP=0, SCALE=True, MARGIN=0):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    # given the predicted price, true price, and an error margin, return true if the predicted price is within the bounds of that error margin
    score = lambda pred_future, true_future, margin: 1 if pred_future >= (true_future - (true_future*margin)) and pred_future <= (true_future + (true_future*margin)) else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    margin_test = data["margin"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))
        margin_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(margin_test, axis=0)))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df["margin"] = margin_test
    test_df[f"Close_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_Close_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # 
    final_df["score"] = list(map(score,
                                final_df[f"Close_{LOOKUP_STEP}"],
                                final_df[f"true_Close_{LOOKUP_STEP}"],
                                final_df["margin"]
    ))
    return final_df

def predict(model, data, N_STEPS=0, SCALE=True):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

def evaluate(model, data, LOSS, SCALE=True, LOOKUP_STEP=0, N_STEPS=0, show_graph=False, MARGIN=0, save_stats=False, TKR=""):
    print(colored('starting model evaluation...', 'blue'))
    # evaluate the model
    loss, mae, mse = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["Close"].inverse_transform([[mae]])[0][0]
        mean_squared_error = data["column_scaler"]["Close"].inverse_transform([[mse]])[0][0]
        # mean_absolute_percentage_error = data["column_scaler"]["Close"].inverse_transform([[mape]])[0][0]
    else:
        mean_absolute_error = mae
        mean_squared_error = mse
        # mean_absolute_percentage_error = mape
    final_df = get_final_df(model, data, LOOKUP_STEP=LOOKUP_STEP, SCALE=SCALE, MARGIN=MARGIN)
    future_price = predict(model, data, N_STEPS=N_STEPS, SCALE=SCALE)
    # printing stats
    original_df = data["df"]
    stats = get_stats(original_df)
    latest_price = original_df["Close"].tail(1).values[0]
    print("Stats on data:", *stats, sep='\n\n')
    print("Stats on model:", sep='\n\n')
    print(f"Loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    # print("Mean Squared Error:", mean_squared_error)
    # print("Mean Absolute Percentage Error", mean_absolute_percentage_error)
    print(f"\nThe model predicts that the future {TKR} price after {LOOKUP_STEP} days will be ${future_price:.2f}")
    print(f"Average error margin: +/- {(mean_absolute_error/latest_price)*100:.2f}%")
    if save_stats:
        file_name = 'output/stats/runResults.csv'
        statsDF = pd.read_csv(file_name)
        statsDF.loc[-1] = [TKR, mean_absolute_error, mean_squared_error, loss, latest_price] # add data from this run
        statsDF.to_csv(file_name, index=False)
    if show_graph:
        plot_graph_price_diff(final_df, LOOKUP_STEP)

def get_stats(df):
    price_min_max_mean = df.groupby('Symbol').agg({'Close': ['min', 'max', 'mean']})
    date_min_max = df.groupby('Symbol').agg({'Date': ['min', 'max']})
    latest_price = df["Close"].tail(1).values[0]
    return [date_min_max, price_min_max_mean, f"Latest closing price in dataset: ${latest_price}"]
