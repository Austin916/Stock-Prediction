## Project Overview

This project aims to identify stocks nearing the end of their accumulation phase, a key point where strategic buying by large investors often occurs. By combining traditional technical analysis with advanced machine learning methods, the system seeks to enhance the accuracy and reliability of detecting strong stocks with significant potential.

Key methodologies include:

- Utilizing technical indicators like Bollinger Bands, RSI, ATR, and others to analyze stock behavior.
- Applying machine learning models, such as Isolation Forest, for anomaly detection.
- Designing a backtesting system to evaluate strategy performance with metrics like the Sharpe Ratio.

## Features

- **Technical Indicator Calculation**:
  - ATR, Bollinger Bands, RSI, MACD, and MFI.
  - Price and volume change rates for feature engineering.
- **Anomaly Detection**:
  - Isolation Forest for identifying anomalies in stock price and volume patterns.
- **Backtesting Framework**:
  - Parameter optimization and performance evaluation.
  - Sharpe Ratio as a primary metric for assessing risk-adjusted returns.
- **Results Visualization**:
  - Scatter plots, stock price trends, and technical indicator charts.

## Technologies

- **Programming Language**: Python
- **Libraries**: 
  - `pandas`, `numpy`, `scikit-learn` for data manipulation and machine learning.
  - `matplotlib`, `seaborn` for visualization.
  - `sqlite3` for storing analysis results.
- **Machine Learning Model**:
  - Isolation Forest for anomaly detection.

## Project Structure

├── final\_project\_Zhehan.pdf &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Final project report  
├── method\_chineseversion.docx &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Initial project methodology  
├── final\_test.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Main Python script for analysis and prediction  
├── all\_stock.zip &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Stock data files  
├── outputs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Results, reports, and visualizations   
└── README.md &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Project documentation    

##  Results

### Example Analysis (Stock 600529)

- **Optimal Parameters**:
  - Minimum Duration: 30 days
  - Maximum Duration: 60 days
  - Price Change Threshold: 15%
  - Volume Change Threshold: 50%
- **Performance Metrics**:
  - Sharpe Ratio: 2.375
  - Average Return: 20.40%
  - Win Rate: 100%
- **Anomaly Statistics**:
  - Total anomalies: 24
  - Average price change: 43%
  - Average volume change: 165%

## Visualization Examples

- **Anomaly Distribution**: Scatter plots highlighting price change, volume change, and duration.
- **Time Series Analysis**: Stock price trends before and after anomalies.


## References

1. Zhang, Y., Chu, G., et al. (2020). *Machine Learning for Stock Prediction Based on Fundamental Analysis*.
2. Ozbayoglu, A. M., et al. (2020). *Deep Learning for Financial Applications*.
3. Lo, A. W., et al. (2000). *Foundations of Technical Analysis*.






