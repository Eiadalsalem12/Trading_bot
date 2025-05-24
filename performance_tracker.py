import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PerformanceTracker:
    def __init__(self):
        # Get API credentials from environment variables
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in .env file. Please make sure BINANCE_API_KEY and BINANCE_API_SECRET are set.")
        
        self.client = Client(api_key, api_secret)
        self.initial_capital = self._get_total_balance()
        self.current_capital = self.initial_capital
        self.trades_history = []
        self.daily_performance = []
        self.backtest_results = []
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
    def _get_total_balance(self):
        """Get total balance in USDT from Binance Futures account."""
        try:
            # Get futures account balance
            futures_account = self.client.futures_account_balance()
            total_balance = 0
            
            for asset in futures_account:
                if asset['asset'] == 'USDT':
                    total_balance = float(asset['balance'])
                    break
            
            return total_balance
        except BinanceAPIException as e:
            print(f"Error getting futures balance: {e}")
            return 0

    def add_trade(self, symbol, side, entry_price, quantity, exit_price=None, exit_time=None):
        """Record a new trade."""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'status': 'open' if exit_price is None else 'closed'
        }
        
        if exit_price is not None:
            trade['pnl'] = self._calculate_pnl(trade)
            trade['pnl_percentage'] = (trade['pnl'] / (entry_price * quantity)) * 100
        
        self.trades_history.append(trade)
        self._update_capital(trade)
        self._save_trade(trade)
        
    def _calculate_pnl(self, trade):
        """Calculate profit/loss for a trade."""
        if trade['side'] == 'BUY':
            return (trade['exit_price'] - trade['entry_price']) * trade['quantity']
        else:
            return (trade['entry_price'] - trade['exit_price']) * trade['quantity']
    
    def _update_capital(self, trade):
        """Update current capital based on trade."""
        if trade.get('pnl') is not None:
            self.current_capital += trade['pnl']
            self.daily_performance.append({
                'timestamp': trade['exit_time'] or datetime.now(),
                'capital': self.current_capital,
                'pnl': trade['pnl'],
                'pnl_percentage': trade.get('pnl_percentage', 0)
            })
    
    def _save_trade(self, trade):
        """Save trade to JSON file."""
        filename = f'data/trades_{datetime.now().strftime("%Y%m%d")}.json'
        
        # Load existing trades if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                trades = json.load(f)
        else:
            trades = []
        
        # Convert datetime objects to strings
        trade_copy = trade.copy()
        trade_copy['timestamp'] = trade_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if trade_copy['exit_time']:
            trade_copy['exit_time'] = trade_copy['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        trades.append(trade_copy)
        
        # Save updated trades
        with open(filename, 'w') as f:
            json.dump(trades, f, indent=4)
    
    def plot_performance(self):
        """Create performance visualization plots."""
        if not self.daily_performance:
            print("No performance data available to plot.")
            return
            
        # Convert daily performance to DataFrame
        df = pd.DataFrame(self.daily_performance)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot capital over time
        ax1.plot(df.index, df['capital'], label='Capital')
        ax1.set_title('Capital Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True)
        
        # Plot PnL percentage
        ax2.bar(df.index, df['pnl_percentage'], label='PnL %')
        ax2.set_title('Daily Profit/Loss Percentage')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('PnL %')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/performance_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
    
    def generate_report(self):
        """Generate performance report."""
        if not self.trades_history:
            return "No trades recorded yet."
        
        closed_trades = [t for t in self.trades_history if t['status'] == 'closed']
        
        report = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_trades': len(closed_trades),
            'winning_trades': len([t for t in closed_trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in closed_trades if t['pnl'] < 0]),
            'total_pnl': sum(t['pnl'] for t in closed_trades),
            'win_rate': len([t for t in closed_trades if t['pnl'] > 0]) / len(closed_trades) * 100 if closed_trades else 0,
            'average_win': np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in closed_trades) else 0,
            'average_loss': np.mean([t['pnl'] for t in closed_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in closed_trades) else 0,
            'largest_win': max([t['pnl'] for t in closed_trades]) if closed_trades else 0,
            'largest_loss': min([t['pnl'] for t in closed_trades]) if closed_trades else 0
        }
        
        # Save report to file
        with open(f'data/report_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report
    
    def add_backtest_result(self, start_date, end_date, initial_capital, final_capital, trades):
        """Add backtest results."""
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': ((final_capital - initial_capital) / initial_capital) * 100,
            'trades': trades
        }
        
        self.backtest_results.append(result)
        
        # Save backtest results
        with open(f'data/backtest_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(self.backtest_results, f, indent=4)
        
        return result

if __name__ == "__main__":
    try:
        # Create a performance tracker instance
        tracker = PerformanceTracker()
        
        print(f"\nInitial Capital: ${tracker.initial_capital:.2f}")
        
        # Generate and print performance report
        report = tracker.generate_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=4))
        
        # Create performance plots only if there's data
        if tracker.daily_performance:
            tracker.plot_performance()
            print("\nPerformance plots have been saved to the 'plots' directory.")
        else:
            print("\nNo trading data available to create performance plots.")
            
    except ValueError as e:
        print(f"Error: {e}")
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}") 