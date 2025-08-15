# -Time-Series-Forecasting-using-REST-API
Time-series data is widely used in domains like finance, weather forecasting, sales analysis,  and healthcare. This project is a **Streamlit-based interactive dashboard** for performing time series forecasting on stock market data.  
It works to:
- Retrieve historical stock prices from Yahoo Finance.
- Perform exploratory data analysis (EDA).
- Apply multiple forecasting models including statistical, machine learning, and deep learning approaches.
- Visualize and compare forecast performance.

  ## Tools & Libraries 
This project uses the following tools and Python libraries:

- **Streamlit** – Interactive web app framework.
- **Pandas** – Data manipulation.
- **NumPy** – Numerical operations.
- **yfinance** – Fetch stock data.
- **Seaborn** & **Matplotlib** – Data visualization.
- **statsmodels** – Statistical modeling (ADF test, ARIMA, SARIMA, ACF plots).
- **scikit-learn** – Machine learning models and preprocessing.
- **XGBoost** – Gradient boosting regression.
- **Keras** – Deep learning LSTM model.
- **Warnings** – Suppressing unnecessary warnings.

## Installation
Install all dependencies using:

```bash
pip install streamlit pandas numpy yfinance seaborn matplotlib statsmodels scikit-learn xgboost keras
```
## Features
1. Data retreival:
    Fetch stock price data from Yahoo Finance.
    Clean, preprocess, and store data in Streamlit's session state.
2. Data exploration:
   Line charts for stock closing prices.
   Monthly heatmaps of average prices.
  Autocorrelation plots.
4. Forecasting models
   ARIMA – Autoregressive Integrated Moving Average.
   SARIMA – Seasonal ARIMA.
   Random Forest – Tree-based regression.
   XGBoost – Gradient boosting regression.
   LSTM – Long Short-Term Memory neural network.
6. Evaluation metrics
   Mean Absolute Error (MAE)
   Root Mean Squared Error (RMSE)
   Mean Absolute Percentage Error (MAPE)
8. Visualization
  Interactive plots comparing actual vs forecasted prices.
  Train/test split visualization.

## How to run
1. Save the script as app.py
2. Run the streamlit app
   ```bash
   streamlit run app.py
3. Open the provided local URL in your browser.
