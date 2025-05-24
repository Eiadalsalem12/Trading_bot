import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StatisticalAnalysis:
    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics."""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    @staticmethod
    def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_volatility if downside_volatility != 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len(returns[returns > 0]) / len(returns),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else float('inf')
        }
    
    @staticmethod
    def analyze_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlation between different assets."""
        return returns_df.corr()
    
    @staticmethod
    def calculate_optimal_position_size(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        win_rate = len(returns[returns > 0]) / len(returns)
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        if avg_loss == 0:
            return 0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(kelly, 0.5))  # Limit to 50% of portfolio
    
    @staticmethod
    def backtest_strategy(
        df: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        initial_capital: float = 10000
    ) -> Dict[str, any]:
        """Backtest a trading strategy."""
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(1, len(df)):
            if entry_signals.iloc[i] and position == 0:
                # Enter position
                position = capital / df['close'].iloc[i]
                entry_price = df['close'].iloc[i]
                entry_time = df.index[i]
                
            elif exit_signals.iloc[i] and position > 0:
                # Exit position
                exit_price = df['close'].iloc[i]
                exit_time = df.index[i]
                pnl = position * (exit_price - entry_price)
                capital += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': pnl / (position * entry_price)
                })
                
                position = 0
        
        # Calculate performance metrics
        returns = pd.Series([trade['return'] for trade in trades])
        metrics = StatisticalAnalysis.calculate_metrics(returns)
        
        return {
            'trades': trades,
            'metrics': metrics,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital
        }
    
    @staticmethod
    def plot_performance(trades: List[Dict], title: str = "Strategy Performance"):
        """Plot strategy performance."""
        # Create equity curve
        equity_curve = pd.Series(
            [trade['pnl'] for trade in trades],
            index=[trade['exit_time'] for trade in trades]
        ).cumsum()
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L')
        plt.grid(True)
        plt.show()
        
        # Plot drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve / rolling_max - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def monte_carlo_simulation(
        returns: pd.Series,
        n_simulations: int = 1000,
        n_days: int = 252
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform Monte Carlo simulation."""
        # Calculate parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate simulations
        simulations = np.random.normal(
            mu,
            sigma,
            (n_simulations, n_days)
        )
        
        # Calculate cumulative returns
        cumulative_returns = (1 + simulations).cumprod(axis=1)
        
        # Calculate confidence intervals
        percentiles = np.percentile(
            cumulative_returns,
            [5, 25, 50, 75, 95],
            axis=0
        )
        
        return cumulative_returns, {
            'mean': np.mean(cumulative_returns, axis=0),
            'std': np.std(cumulative_returns, axis=0),
            'percentiles': percentiles
        } 