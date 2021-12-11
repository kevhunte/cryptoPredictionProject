import matplotlib.pyplot as plt
import numpy as np

def plot_graph(test_df, LOOKUP_STEP=0):
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

def evaluate(model, data, LOSS, SCALE=True, LOOKUP_STEP=0, N_STEPS=0, show_graph=False, MARGIN=0, TKR=""):
    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["Close"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
    final_df = get_final_df(model, data, LOOKUP_STEP=LOOKUP_STEP, SCALE=SCALE, MARGIN=MARGIN)
    future_price = predict(model, data, N_STEPS=N_STEPS, SCALE=SCALE)
    # Evaluation Stats
    # we calculate the accuracy by counting the number of positive profits
    # accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    accuracy_score = (len(final_df[final_df['score'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    # printing stats
    original_df = data["df"]
    stats = get_stats(original_df)
    print("Stats on data:", *stats, sep='\n\n')
    print(f"Future {TKR} price after {LOOKUP_STEP} days is ${future_price:.2f}")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print(f"Accuracy score of predictions within {(1-MARGIN):.2f}%:", accuracy_score)
    if show_graph:
        plot_graph(final_df, LOOKUP_STEP)

def get_stats(df):
    price_min_max_mean = df.groupby('Symbol').agg({'Close': ['min', 'max', 'mean']})
    date_min_max = df.groupby('Symbol').agg({'Date': ['min', 'max']})
    return [date_min_max, price_min_max_mean]
