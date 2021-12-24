# Crypto Predictor Project

Built using RNN and LSTM

## Usage

download the dataset from Kaggle and place the files in a folder named ```cryptoData```

```sh
python3 main.py '<insert supported ticker name here>'
```

generating the relative absolute error

```sh
python3 main.py --plot-stats
```

viewing the price data and additional metrics

```sh
python3 main.py --plot-price
```
### Dataset from Kaggle

[Kaggle](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory)

### commands

## Testing


## Customization

The config.json holds all parameters to the functions. Altering the values there will change the output without having to modify any code