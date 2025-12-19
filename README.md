# EUR/PLN Forex Prediction with LSTM

This project implements a Deep Learning model to forecast the direction of the EUR/PLN currency pair. 

### Key Features:
* **Architecture:** LSTM (Long Short-Term Memory) network built with PyTorch.
* **Data Source:** Automated data fetching via `yfinance`.
* **Data Integrity:** Implements a strict 91-day gap between training and testing sets to prevent data leakage.
* **Classification:** Predicts binary outcomes (Price Increase vs. Price Decrease) based on a 64-day historical window.
* **Preprocessing:** Includes MinMaxScaler and time-series sequencing.

### Tech Stack:
* Python
* PyTorch
* Pandas/NumPy
* Scikit-learn
