import sys, json, os
import pandas as pd
from utils.prep_utils import load_data
from utils.create_utils import create_model
from utils.train_utils import train_model
from utils.test_utils import evaluate, plot_graph_run_stats, plot_graph_price

"""
ticker => name of the particular coin to generate a model for
run_settings => tells the handler how to run
run_all => pass in to generate for all coins
"""
def handler (ticker, run_settings, run_all=False):
    # prep test data for the model with customized configs
    load_data_params = { "ticker": ticker, "MARGIN": run_settings["test"]["MARGIN"], **run_settings["prep"] }
    data = load_data(**load_data_params)
    if data is None:
        print('loading data was unsuccessful')
        return
    # create model
    create_model_params = { "n_features": len(run_settings["prep"]["feature_columns"]), **run_settings["create"]}
    model, model_name = create_model(**create_model_params)
    # train model
    full_model_name = f'{ticker}-{model_name}' # include ticker trained with
    training_params = { "data": data, "model": model, "model_name": full_model_name, **run_settings["train"] }
    train_model(**training_params)
    # test model
    model_path = os.path.join(run_settings["train"]["output_dir"],"results", full_model_name) + ".h5"
    model.load_weights(model_path) # optimal weights from training
    evaluate_params = { "model": model, "data": data, "TKR": ticker, **run_settings["test"] }
    evaluate(**evaluate_params)

if __name__ == '__main__':
    # grab whatever cmd line args and pass into handler.
    coin = sys.argv[1] if len(sys.argv) > 1 else ''
    run_all = True if '--run-all' in sys.argv else False
    plot_stats = True if '--plot-stats' in sys.argv else False
    plot_price = True if '--plot-price' in sys.argv else False
    with open('config.json','r') as f:
        run_settings = json.load(f)
    if plot_stats:
        file_name = 'output/stats/runResults.csv'
        runResultsDF = pd.read_csv(file_name)
        plot_graph_run_stats(runResultsDF)
    elif plot_price:
        file_name = 'cryptoData/coin_Bitcoin.csv'
        priceDF = pd.read_csv(file_name)
        file_name = 'cryptoData/coin_Ethereum.csv'
        priceDF = pd.concat([priceDF, pd.read_csv(file_name)]) # merge
        file_name = 'cryptoData/coin_Cardano.csv'
        priceDF = pd.concat([priceDF, pd.read_csv(file_name)]).reset_index(drop=True) # merge
        plot_graph_price(priceDF)
    else:
        for _ in range(10):
        # automate eval metrics
            handler(coin, run_settings, run_all=run_all)