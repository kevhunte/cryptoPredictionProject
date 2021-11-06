
# root of the project. This is the file that should get run 
import sys, json

"""
ticker => name of the particular coin to generate a model for
run_settings => tells the handler how to run
run_all => pass in to generate for all coins
"""
def handler (ticker, run_settings, run_all=False):
    print('starting up...', ticker, run_settings, run_all, sep='\n')
    # prep data 
    # create model 
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