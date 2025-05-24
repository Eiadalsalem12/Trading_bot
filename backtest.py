import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collector import DataCollector
from model import PricePredictionModel
from performance_tracker import PerformanceTracker
from config import TRADING_PAIRS, CAPITAL_PERCENTAGE, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE

class Backtester:
    def __init__(self, client, initial_capital=30, config=None):
        self.client = client
        self.data_collector = DataCollector(client, config if config is not None else {})
        self.performance_tracker = PerformanceTracker(initial_capital)
        self.models = {}
        
    def initialize_models(self, start_date, end_date):
        """Initialize and train models for backtesting."""
        for symbol in TRADING_PAIRS:
            df = self.data_collector.get_historical_klines(symbol, '1h', start_time=start_date, end_time=end_date)
            df = self.data_collector.add_technical_indicators(df)
            X, y = self.data_collector.prepare_training_data(df)
            model = PricePredictionModel(input_shape=(X.shape[1],))
            model.train(X.values, y.values)
            self.models[symbol] = model
    
    def run_backtest(self, start_date, end_date):
        """Run backtest simulation."""
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize models
        self.initialize_models(start_date, end_date)
        
        # Get historical data
        for symbol in TRADING_PAIRS:
            df = self.data_collector.get_historical_klines(symbol, '1h', start_time=start_date, end_time=end_date)
            df = self.data_collector.add_technical_indicators(df)
            
            # Simulate trading
            position = None
            entry_price = None
            quantity = None
            
            for i in range(24, len(df)):
                current_data = df.iloc[i-24:i]
                current_price = df.iloc[i]['close']
                
                # Make prediction
                model = self.models[symbol]
                prediction = model.predict(current_data.values)
                
                # Trading logic
                if position is None:  # No open position
                    if prediction is not None and prediction > 0.001:  # Buy signal
                        quantity = self._calculate_position_size(current_price)
                        if quantity > 0:
                            position = 'BUY'
                            entry_price = current_price
                            self.performance_tracker.add_trade(
                                symbol, position, entry_price, quantity
                            )
                    
                    elif prediction is not None and prediction < -0.001:  # Sell signal
                        quantity = self._calculate_position_size(current_price)
                        if quantity > 0:
                            position = 'SELL'
                            entry_price = current_price
                            self.performance_tracker.add_trade(
                                symbol, position, entry_price, quantity
                            )
                
                else:  # Check exit conditions
                    if position == 'BUY':
                        # Check stop loss
                        if current_price <= entry_price * (1 - STOP_LOSS_PERCENTAGE):
                            self._close_position(symbol, position, entry_price, quantity, current_price, df.index[i])
                            position = None
                        
                        # Check take profit
                        elif current_price >= entry_price * (1 + TAKE_PROFIT_PERCENTAGE):
                            self._close_position(symbol, position, entry_price, quantity, current_price, df.index[i])
                            position = None
                    
                    elif position == 'SELL':
                        # Check stop loss
                        if current_price >= entry_price * (1 + STOP_LOSS_PERCENTAGE):
                            self._close_position(symbol, position, entry_price, quantity, current_price, df.index[i])
                            position = None
                        
                        # Check take profit
                        elif current_price <= entry_price * (1 - TAKE_PROFIT_PERCENTAGE):
                            self._close_position(symbol, position, entry_price, quantity, current_price, df.index[i])
                            position = None
        
        # Generate performance report
        report = self.performance_tracker.generate_report()
        self.performance_tracker.plot_performance()
        
        # Save backtest results
        self.performance_tracker.add_backtest_result(
            start_date,
            end_date,
            self.performance_tracker.initial_capital,
            self.performance_tracker.current_capital,
            self.performance_tracker.trades_history
        )
        
        return report
    
    def _calculate_position_size(self, price):
        """Calculate position size based on risk management rules."""
        available_balance = self.performance_tracker.current_capital * CAPITAL_PERCENTAGE
        risk_amount = available_balance * STOP_LOSS_PERCENTAGE
        position_size = risk_amount / (price * STOP_LOSS_PERCENTAGE)
        
        return min(position_size, available_balance / price)
    
    def _close_position(self, symbol, position, entry_price, quantity, exit_price, exit_time):
        """Close a position and record the trade."""
        self.performance_tracker.add_trade(
            symbol, position, entry_price, quantity, exit_price, exit_time
        )

if __name__ == "__main__":
    from binance.client import Client
    from dotenv import load_dotenv
    import os
    
    # Load API credentials
    load_dotenv()
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    
    # Run backtest
    backtester = Backtester(client, initial_capital=30)
    start_date = datetime.now() - timedelta(days=30)  # Last 30 days
    end_date = datetime.now()
    
    report = backtester.run_backtest(start_date, end_date)
    print("\nBacktest Results:")
    print(f"Initial Capital: ${report['initial_capital']:.2f}")
    print(f"Final Capital: ${report['current_capital']:.2f}")
    print(f"Total Return: {((report['current_capital'] - report['initial_capital']) / report['initial_capital'] * 100):.2f}%")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']:.2f}%")
    print(f"Average Win: ${report['average_win']:.2f}")
    print(f"Average Loss: ${report['average_loss']:.2f}")
    print(f"Largest Win: ${report['largest_win']:.2f}")
    print(f"Largest Loss: ${report['largest_loss']:.2f}") 