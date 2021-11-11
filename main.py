import sys, json
from utils.prep_utils import load_data
from utils.create_utils import create_model

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
    model = create_model(**run_settings["create"])
    # train model
    # test model
    pass

if __name__ == '__main__':
    # grab whatever cmd line args and pass into handler.
    coin = sys.argv[1] if len(sys.argv) > 1 else ''
    run_all = True if '--run-all' in sys.argv else False
    with open('config.json','r') as f:
        run_settings = json.load(f)
    handler(coin, run_settings, run_all=run_all)