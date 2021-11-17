import sys, json
from utils.prep_utils import load_data
from utils.create_utils import create_model
from utils.train_utils import train_model

"""
ticker => name of the particular coin to generate a model for
run_settings => tells the handler how to run
run_all => pass in to generate for all coins
"""
def handler (ticker, run_settings, run_all=False):
    # prep test data for the model with customized configs
    load_data_params = { "ticker": ticker, **run_settings["prep"] }
    data = load_data(**load_data_params)
    if data is None:
        print('loading data was unsuccessful')
        return
    # create model
    model, model_name = create_model(**run_settings["create"])
    # train model
    training_params = { "data": data, "model": model, "model_name": f'{ticker}-{model_name}', **run_settings["train"] }
    train_model(**training_params)
    # test model
    pass

if __name__ == '__main__':
    # grab whatever cmd line args and pass into handler.
    coin = sys.argv[1] if len(sys.argv) > 1 else ''
    run_all = True if '--run-all' in sys.argv else False
    with open('config.json','r') as f:
        run_settings = json.load(f)
    handler(coin, run_settings, run_all=run_all)