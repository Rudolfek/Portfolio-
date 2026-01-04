import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset #do koncowego przygotowania danych
import pysam
import yfinance as yf #do danych finansowych

from sklearn.preprocessing import MinMaxScaler #do skalowania

from sklearn.metrics import classification_report #do robienia raportów o uczeniu

import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm

import optuna
from optuna.trial import TrialState

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def funkcja_do_danych(data):
    data=data.copy()
    data=pd.DataFrame(data)
    data=data.reset_index()
    data["Date"]=pd.to_datetime(data["Date"])
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['log_vol_change'] = np.log((data['Volume'] + 1e-6) / (data['Volume'].shift(1) + 1e-6))
    data['high_low_diff'] = (data['High'] - data['Low']) / data['Close']

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6) # Zabezpieczenie przed dzieleniem przez 0
    data['rsi'] = 100 - (100 / (1 + rs))
    # Normalizujemy RSI do zakresu 0-1 (Autoencoder to lubi)
    data['rsi'] = data['rsi'] / 100.0

    sma_window = 50
    sma = data['Close'].rolling(window=sma_window).mean()
    data['dist_sma'] = (data['Close'] - sma) / sma

    data['volatility'] = data['log_ret'].rolling(window=14).std()
    data=data.dropna()
   
    features = [
        'log_ret', 
        'log_vol_change', 
        'high_low_diff', 
        'rsi', 
        'dist_sma', 
        'volatility'
    ]
    
    # Zwracamy tylko wybrane kolumny
    return data[features].astype('float32')



def sekwencje(data_x, data_y, seq_len):
    if isinstance(data_x, torch.Tensor): data_x = data_x.numpy()
    if isinstance(data_y, torch.Tensor): data_y = data_y.numpy()
    x_view = np.lib.stride_tricks.sliding_window_view(data_x, window_shape=seq_len, axis=0)
    x_seq = x_view.swapaxes(1, 2)
    y_seq = data_y[seq_len-1 : len(data_x)] 
    min_len = min(len(x_seq), len(y_seq))
    return x_seq[:min_len], y_seq[:min_len]
