import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class FinancialAnalysis:
    @staticmethod
    def calculate_portfolio_metrics(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns.index)
            for symbol, weight in weights.items():
                portfolio_returns += returns[symbol] * weight
            
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
            }
        except Exception as e:
            logging.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, any]:
        """Optimize portfolio weights using Modern Portfolio Theory."""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Generate random portfolios
            num_portfolios = 10000
            results = np.zeros((num_portfolios, len(returns.columns) + 2))
            
            for i in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(len(returns.columns))
                weights = weights / np.sum(weights)
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
                
                # Store results
                results[i, 0] = portfolio_std
                results[i, 1] = portfolio_return
                results[i, 2:] = weights
            
            # Find optimal portfolio
            optimal_idx = np.argmax(results[:, 1] / results[:, 0])
            optimal_weights = dict(zip(returns.columns, results[optimal_idx, 2:]))
            
            return {
                'optimal_weights': optimal_weights,
                'expected_return': results[optimal_idx, 1],
                'expected_volatility': results[optimal_idx, 0],
                'sharpe_ratio': results[optimal_idx, 1] / results[optimal_idx, 0]
            }
        except Exception as e:
            logging.error(f"Error optimizing portfolio: {e}")
            return {}
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate various risk metrics."""
        try:
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (ES)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Tail risk metrics
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            return {
                'volatility': volatility,
                'downside_volatility': downside_volatility,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}
    
    @staticmethod
    def analyze_correlation(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze correlation between assets."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Calculate p-values
            p_values = pd.DataFrame(
                np.zeros_like(corr_matrix),
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )
            
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if i != j:
                        corr, p_value = stats.pearsonr(
                            returns[corr_matrix.columns[i]],
                            returns[corr_matrix.columns[j]]
                        )
                        p_values.iloc[i, j] = p_value
            
            return {
                'correlation': corr_matrix,
                'p_values': p_values
            }
        except Exception as e:
            logging.error(f"Error analyzing correlation: {e}")
            return {}
    
    @staticmethod
    def calculate_position_sizing(
        capital: float,
        risk_per_trade: float,
        stop_loss: float,
        volatility: float
    ) -> float:
        """Calculate position size based on risk management rules."""
        try:
            # Calculate risk amount
            risk_amount = capital * risk_per_trade
            
            # Adjust for volatility
            volatility_factor = max(0.5, 1 - volatility)
            
            # Calculate position size
            position_size = (risk_amount / stop_loss) * volatility_factor
            
            return min(position_size, capital * 0.5)  # Limit to 50% of capital
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.0
    
    @staticmethod
    def analyze_drawdown(returns: pd.Series) -> Dict[str, any]:
        """Analyze drawdown characteristics."""
        try:
            # Calculate drawdowns
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            
            # Calculate drawdown metrics
            max_drawdown = drawdowns.min()
            avg_drawdown = drawdowns[drawdowns < 0].mean()
            drawdown_duration = (drawdowns < 0).astype(int).groupby(
                (drawdowns < 0).astype(int).cumsum()
            ).cumsum()
            max_drawdown_duration = drawdown_duration.max()
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'drawdowns': drawdowns
            }
        except Exception as e:
            logging.error(f"Error analyzing drawdown: {e}")
            return {}
    
    @staticmethod
    def plot_portfolio_analysis(returns: pd.DataFrame, weights: Dict[str, float], title: str = "Portfolio Analysis"):
        """Plot portfolio analysis results."""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot cumulative returns
            plt.subplot(2, 2, 1)
            cumulative_returns = (1 + returns).cumprod()
            for symbol in returns.columns:
                plt.plot(cumulative_returns[symbol], label=symbol)
            plt.title('Cumulative Returns')
            plt.legend()
            
            # Plot drawdowns
            plt.subplot(2, 2, 2)
            for symbol in returns.columns:
                returns_series = returns[symbol]
                cumulative = (1 + returns_series).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdowns = cumulative / rolling_max - 1
                plt.plot(drawdowns, label=symbol)
            plt.title('Drawdowns')
            plt.legend()
            
            # Plot rolling volatility
            plt.subplot(2, 2, 3)
            for symbol in returns.columns:
                rolling_vol = returns[symbol].rolling(window=20).std() * np.sqrt(252)
                plt.plot(rolling_vol, label=symbol)
            plt.title('Rolling Volatility (20-day)')
            plt.legend()
            
            # Plot portfolio weights
            plt.subplot(2, 2, 4)
            plt.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
            plt.title('Portfolio Weights')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting portfolio analysis: {e}")

if __name__ == "__main__":
    print("تم تشغيل ملف financial_analysis.py بنجاح") 