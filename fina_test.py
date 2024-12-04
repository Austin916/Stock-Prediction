import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import json


# 指定根目录
root_dir = 'all_stock'
folders = ['Chemical_industry', 'Medical_treatment', 'Property']

def calculate_atr(data, period=14):
    high_low = data['最高价'] - data['最低价']
    high_close = abs(data['最高价'] - data['收盘价'].shift())
    low_close = abs(data['最低价'] - data['收盘价'].shift())
    tr = high_low.to_frame('tr').join(high_close.to_frame('hc')).join(low_close.to_frame('lc')).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['收盘价'].rolling(window=window).mean()
    rolling_std = data['收盘价'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_rsi(data, window=14):
    delta = data['收盘价'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_moving_average(data, window=20):
    volume_ma = data['成交量(手)'].rolling(window=window).mean()
    return volume_ma

def calculate_macd(data):
    short_ema = data['收盘价'].ewm(span=12).mean()
    long_ema = data['收盘价'].ewm(span=26).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=9).mean()
    return macd, macd_signal

def calculate_mfi(data, period=14):
    typical_price = (data['最高价'] + data['最低价'] + data['收盘价']) / 3
    money_flow = typical_price * data['成交量(手)']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi

def read_and_preprocess_data(file):
    data = pd.read_csv(file)
    required_columns = ['最高价', '最低价', '收盘价', '成交量(手)', '交易日期']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"File: {file}")
        print(f"Missing required columns: {', '.join(missing_columns)}")
        print("Skipping this file.")
        print("---")
        return None

    data['交易日期'] = pd.to_datetime(data['交易日期'], format='%Y%m%d')
    data.sort_values('交易日期', ascending=True, inplace=True)
    
    # 计算技术指标
    data['ATR'] = calculate_atr(data)
    data['BollingerUpper'], data['BollingerLower'] = calculate_bollinger_bands(data)
    data['RSI'] = calculate_rsi(data)
    data['VolumeMA'] = calculate_volume_moving_average(data)
    data['MACD'], data['MACDSignal'] = calculate_macd(data)
    data['MFI'] = calculate_mfi(data)
    
    # 添加新特征
    data['PriceChangeRate'] = data['收盘价'].pct_change()
    data['VolumeChangeRate'] = data['成交量(手)'].pct_change()
    
    data.dropna(inplace=True)

    # 进行异常检测
    data, _ = anomaly_detection(data)

    return data

def anomaly_detection(data):
    features = ['ATR', 'RSI', 'VolumeMA', 'MACD', 'MACDSignal', 'MFI', 'PriceChangeRate', 'VolumeChangeRate']
    X = data[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # IsolationForest
    model = IsolationForest(contamination=0.1, random_state=42)  # 使用固定参数而不是GridSearchCV以提高速度
    anomalies = model.fit_predict(X_scaled)
    
    # 将结果添加到数据中
    data['Anomaly'] = anomalies
    data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})  # 将-1映射为1（异常），1映射为0（正常）
    
    return data, model

def identify_peaks(data, window=20):
    data['Rolling_Max'] = data['收盘价'].rolling(window=window, center=True).max()
    data['is_peak'] = ((data['收盘价'] == data['Rolling_Max']) & (data['收盘价'] > data['收盘价'].shift(1)) & (data['收盘价'] > data['收盘价'].shift(-1))).astype(int)
    print(f"Number of peaks identified: {data['is_peak'].sum()}")
    return data

def calculate_intervals(data, max_days=30):
    anomaly_starts = data[data['Anomaly'].diff() == 1].index
    peaks_after_anomaly = []
    for start in anomaly_starts:
        start_date = data.loc[start, '交易日期']
        end_date = start_date + pd.Timedelta(days=max_days)
        peak_in_range = data[(data['交易日期'] > start_date) & (data['交易日期'] <= end_date)]['收盘价'].idxmax()
        if peak_in_range > start and data.loc[peak_in_range, 'is_peak'] == 1:
            peaks_after_anomaly.append((start, peak_in_range))
    
    intervals = [(data.loc[peak[1], '交易日期'] - data.loc[peak[0], '交易日期']).days for peak in peaks_after_anomaly]
    avg_interval = np.mean(intervals) if intervals else np.nan
    print(f"Number of valid anomaly-peak pairs: {len(intervals)}")
    return avg_interval, intervals

def visualize_results(data):
    plt.figure(figsize=(15, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(data['交易日期'], data['收盘价'])
    plt.title('Stock Price')
    
    plt.subplot(4, 1, 2)
    plt.plot(data['交易日期'], data['RSI'])
    plt.title('RSI')
    
    plt.subplot(4, 1, 3)
    plt.scatter(data['交易日期'], data['收盘价'], c=data['Anomaly'], cmap='viridis')
    plt.title('Anomalies')
    
    plt.subplot(4, 1, 4)
    plt.scatter(data['交易日期'][data['is_peak'] == 1], data['收盘价'][data['is_peak'] == 1], color='red')
    plt.title('Peaks')
    
    plt.tight_layout()
    plt.show()








# 第二部分：筛选异常波动和评估策略

def filter_anomalies(data, min_duration=30, max_duration=90, price_change_threshold=0.1, volume_change_threshold=1.0):
    data = data.copy()
    data['PricePeak'] = data.groupby((data['Anomaly'] != data['Anomaly'].shift(1)).cumsum())['最高价'].transform('max')
    data['PricePeakDate'] = data.groupby((data['Anomaly'] != data['Anomaly'].shift(1)).cumsum())['交易日期'].transform('last')
    data['StartDate'] = data['交易日期']
    data['Duration'] = (data['PricePeakDate'] - data['StartDate']).dt.days
    
    # 计算价格变化和成交量变化
    data['PriceChange'] = (data['PricePeak'] - data['收盘价']) / data['收盘价']
    data['VolumeChange'] = data['成交量(手)'] / data['成交量(手)'].rolling(window=20).mean()
    
    # 打印调试信息
    print(f"异常总数: {data['Anomaly'].sum()}")
    print(f"持续时间在范围内的异常数: {((data['Duration'] >= min_duration) & (data['Duration'] <= max_duration)).sum()}")
    print(f"价格变化达标的异常数: {(data['PriceChange'] >= price_change_threshold).sum()}")
    print(f"成交量变化达标的异常数: {(data['VolumeChange'] >= volume_change_threshold).sum()}")
    
    # 应用筛选条件
    anomalies = data[
        (data['Anomaly'] == 1) & 
        (data['Duration'] >= min_duration) & 
        (data['Duration'] <= max_duration) &
        (data['PriceChange'] >= price_change_threshold) &
        (data['VolumeChange'] >= volume_change_threshold)
    ]
    
    return anomalies



def evaluate_strategy(anomalies, data, holding_period=30):
    results = []
    for _, anomaly in anomalies.iterrows():
        start_date = anomaly['交易日期']
        end_date = start_date + pd.Timedelta(days=holding_period)
        
        start_price = anomaly['收盘价']
        end_price = data[(data['交易日期'] > start_date) & (data['交易日期'] <= end_date)]['收盘价'].iloc[-1] if len(data[(data['交易日期'] > start_date) & (data['交易日期'] <= end_date)]) > 0 else start_price
        
        returns = (end_price - start_price) / start_price
        results.append(returns)
    
    if not results:
        return {
            'avg_return': np.nan,
            'sharpe_ratio': np.nan,
            'win_rate': np.nan,
            'profit_factor': np.nan
        }
    
    avg_return = np.mean(results)
    sharpe_ratio = np.mean(results) / np.std(results) if np.std(results) != 0 else np.nan
    win_rate = np.sum(np.array(results) > 0) / len(results)
    profit_factor = np.sum(np.array(results)[np.array(results) > 0]) / abs(np.sum(np.array(results)[np.array(results) < 0])) if np.sum(np.array(results) < 0) != 0 else np.inf
    
    return {
        'avg_return': avg_return,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

def backtest(data, min_duration_range, max_duration_range, price_change_threshold_range, volume_change_threshold_range, holding_period=30):
    best_params = None
    best_sharpe = -np.inf
    
    for min_duration in min_duration_range:
        for max_duration in max_duration_range:
            for price_change_threshold in price_change_threshold_range:
                for volume_change_threshold in volume_change_threshold_range:
                    anomalies = filter_anomalies(data, min_duration, max_duration, price_change_threshold, volume_change_threshold)
                    print(f"参数: min_duration={min_duration}, max_duration={max_duration}, "
                          f"price_change_threshold={price_change_threshold}, volume_change_threshold={volume_change_threshold}")
                    print(f"筛选出的异常数量: {len(anomalies)}")
                    
                    if len(anomalies) > 0:
                        results = evaluate_strategy(anomalies, data, holding_period)
                        print(f"夏普比率: {results['sharpe_ratio']}")
                        
                        if results['sharpe_ratio'] > best_sharpe:
                            best_sharpe = results['sharpe_ratio']
                            best_params = {
                                'min_duration': min_duration,
                                'max_duration': max_duration,
                                'price_change_threshold': price_change_threshold,
                                'volume_change_threshold': volume_change_threshold,
                                'sharpe_ratio': results['sharpe_ratio'],
                                'avg_return': results['avg_return'],
                                'win_rate': results['win_rate'],
                                'profit_factor': results['profit_factor']
                            }
                    else:
                        print("没有异常被筛选出来")
                    print("---")
    
    if best_params is None:
        print("使用更宽松的条件重新尝试...")
        for min_duration in min_duration_range:
            for max_duration in max_duration_range:
                for price_change_threshold in price_change_threshold_range:
                    for volume_change_threshold in [0.3, 0.4, 0.5]:  # 尝试更低的成交量变化阈值
                        anomalies = filter_anomalies(data, min_duration, max_duration, price_change_threshold, volume_change_threshold)
                        if len(anomalies) > 0:
                            results = evaluate_strategy(anomalies, data, holding_period)
                            if results['sharpe_ratio'] > best_sharpe:
                                best_sharpe = results['sharpe_ratio']
                                best_params = {
                                    'min_duration': min_duration,
                                    'max_duration': max_duration,
                                    'price_change_threshold': price_change_threshold,
                                    'volume_change_threshold': volume_change_threshold,
                                    'sharpe_ratio': results['sharpe_ratio'],
                                    'avg_return': results['avg_return'],
                                    'win_rate': results['win_rate'],
                                    'profit_factor': results['profit_factor']
                                }
    
    return best_params






# 第三部分：整体结果的呈现
def store_results(data, anomalies, best_params, db_name='stock_analysis.db'):
    conn = sqlite3.connect(db_name)
    
    # 存储异常数据
    anomalies.to_sql('anomalies', conn, if_exists='replace', index=False)
    
    # 将 best_params 转换为 JSON 字符串
    best_params_json = json.dumps(best_params)
    
    # 存储最佳参数
    pd.DataFrame({'best_params': [best_params_json]}).to_sql('best_params', conn, if_exists='replace', index=False)
    
    # 存储原始数据的一些统计信息
    data_stats = pd.DataFrame({
        'total_rows': [len(data)],
        'date_range': [f"{data['交易日期'].min()} to {data['交易日期'].max()}"],
        'average_price': [data['收盘价'].mean()],
        'average_volume': [data['成交量(手)'].mean()],
    })
    data_stats.to_sql('data_stats', conn, if_exists='replace', index=False)
    
    conn.close()

def cross_stock_analysis(db_name='stock_analysis.db'):
    conn = sqlite3.connect(db_name)
    anomalies = pd.read_sql('SELECT * FROM anomalies', conn)
    best_params_json = pd.read_sql('SELECT * FROM best_params', conn)['best_params'][0]
    best_params = json.loads(best_params_json)
    data_stats = pd.read_sql('SELECT * FROM data_stats', conn)
    conn.close()
    
    # 这里可以添加跨股票的分析代码
    # 例如，比较不同股票的异常特征
    
    return anomalies, best_params, data_stats

def generate_report(anomalies, best_params, data_stats, stock_name):
    plt.figure(figsize=(15, 10))
    
    # 绘制异常分布
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=anomalies, x='PriceChange', y='VolumeChange', hue='Duration')
    plt.title(f'Anomaly Distribution - {stock_name}')
    
    # 绘制异常发生的时间序列
    plt.subplot(2, 2, 2)
    plt.plot(anomalies['交易日期'], anomalies['收盘价'], 'ro')
    plt.title(f'Anomaly Occurrences Over Time - {stock_name}')
    
    plt.tight_layout()
    plt.savefig(f'analysis_report_{stock_name}.png')
    plt.close()
    
    # 生成文本报告
    with open(f'analysis_report_{stock_name}.txt', 'w') as f:
        f.write(f"Analysis Report for {stock_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write(f"Data Statistics:\n{data_stats.to_string()}\n\n")
        f.write(f"Total Anomalies: {len(anomalies)}\n")
        f.write(f"Average Price Change: {anomalies['PriceChange'].mean():.2f}\n")
        f.write(f"Average Volume Change: {anomalies['VolumeChange'].mean():.2f}\n")

# 主循环

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file in csv_files:
        print(f"\n处理文件: {file}")
        data = read_and_preprocess_data(file)
        if data is not None:
            print("\n开始回测和策略评估:")
            
            # 检查原始异常
            original_anomalies = data[data['Anomaly'] == 1]
            print(f"原始检测到的异常数量: {len(original_anomalies)}")
            if len(original_anomalies) > 0:
                print("异常示例:")
                print(original_anomalies.head())
            
            min_duration_range = [10, 20, 30]
            max_duration_range = [60, 90, 120]
            price_change_threshold_range = [0.05, 0.1, 0.15]
            volume_change_threshold_range = [0.5, 1.0, 1.5]
            
            best_params = backtest(data, min_duration_range, max_duration_range, 
                                   price_change_threshold_range, volume_change_threshold_range)
            
            if best_params is not None:
                final_anomalies = filter_anomalies(data, best_params['min_duration'], best_params['max_duration'], 
                                                   best_params['price_change_threshold'], best_params['volume_change_threshold'])
                print(f"\n最终筛选出的异常数量: {len(final_anomalies)}")
                if len(final_anomalies) > 0:
                    print("\n异常示例:")
                    print(final_anomalies[['交易日期', '收盘价', 'PriceChange', 'VolumeChange', 'Duration']].head())
                    
                    # 计算这些异常的平均特征
                    print("\n筛选出的异常平均特征:")
                    print(f"平均价格变化: {final_anomalies['PriceChange'].mean():.2f}")
                    print(f"平均成交量变化: {final_anomalies['VolumeChange'].mean():.2f}")
                    print(f"平均持续时间: {final_anomalies['Duration'].mean():.2f} 天")
                    
                    # 添加更多统计信息
                    print("\n异常分布情况:")
                    print(f"价格变化范围: {final_anomalies['PriceChange'].min():.2f} 到 {final_anomalies['PriceChange'].max():.2f}")
                    print(f"成交量变化范围: {final_anomalies['VolumeChange'].min():.2f} 到 {final_anomalies['VolumeChange'].max():.2f}")
                    print(f"持续时间范围: {final_anomalies['Duration'].min():.0f} 到 {final_anomalies['Duration'].max():.0f} 天")
                    
                    # 可视化异常
                    plt.figure(figsize=(12, 8))
                    plt.scatter(final_anomalies['PriceChange'], final_anomalies['VolumeChange'], 
                                c=final_anomalies['Duration'], cmap='viridis')
                    plt.colorbar(label='Duration (days)')
                    plt.xlabel('Price Change')
                    plt.ylabel('Volume Change')
                    plt.title(f'Anomalies Distribution - {os.path.basename(file)}')
                    plt.savefig(f'anomalies_distribution_{os.path.basename(file)}.png')
                    plt.close()
                    
                    # 分析异常前后的股价走势
                    for idx, anomaly in final_anomalies.head().iterrows():
                        start_date = anomaly['交易日期'] - pd.Timedelta(days=30)
                        end_date = anomaly['交易日期'] + pd.Timedelta(days=30)
                        anomaly_period = data[(data['交易日期'] >= start_date) & (data['交易日期'] <= end_date)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(anomaly_period['交易日期'], anomaly_period['收盘价'])
                        plt.axvline(x=anomaly['交易日期'], color='r', linestyle='--')
                        plt.title(f"Stock Price Around Anomaly on {anomaly['交易日期'].date()} - {os.path.basename(file)}")
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.savefig(f'anomaly_price_{os.path.basename(file)}_{idx}.png')
                        plt.close()
                    
                    # 第三部分：存储结果和生成报告
                    stock_name = os.path.splitext(os.path.basename(file))[0]
                    db_name = f'{stock_name}_analysis.db'
                    store_results(data, final_anomalies, best_params, db_name)
                    anomalies, best_params, data_stats = cross_stock_analysis(db_name)
                    generate_report(anomalies, best_params, data_stats, stock_name)
                    print(f"分析报告已生成，请查看 analysis_report_{stock_name}.txt 和 相关的 .png 文件")
                
            else:
                print("未找到有效的参数组合，无法进行进一步分析。")
        
        print("\n" + "="*50 + "\n")  # 添加分隔线

print("所有样本处理完成。")