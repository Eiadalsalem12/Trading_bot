import os
from datetime import datetime, time
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
import json
from typing import Dict, List, Optional
import time as time_module
from data_collector import DataCollector
from model import PricePredictionModel
from config import (
    TRADING_PAIRS, CAPITAL_PERCENTAGE, DAILY_PROFIT_TARGET,
    MAX_LOSS_PERCENTAGE, TRADING_START_TIME, TRADING_END_TIME,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, MAX_OPEN_TRADES,
    MAX_DRAWDOWN_PERCENTAGE, DAILY_LOSS_LIMIT, MAX_RETRIES, RETRY_DELAY,
    API_KEY, API_SECRET, validate_config, PREDICTION_TIMEFRAME, LOOKBACK_PERIOD
)
from tradingview_ta import TA_Handler, Interval

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class Trade:
    def __init__(self, order_id: str, symbol: str, side: str, quantity: float, entry_price: float, trading_mode: str, leverage: int = 1):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.status = 'OPEN'
        self.pnl = 0.0
        self.pnl_percentage = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.risk_reward_ratio = 2.0  # Default risk:reward ratio
        self.trailing_stop = None
        self.trailing_stop_activation = 0.02  # 2% profit to activate trailing stop
        self.trailing_stop_distance = 0.01  # 1% trailing distance
        self.risk_amount = 0.0  # Amount at risk in USDT
        self.position_size = 0.0  # Position size in USDT
        self.leverage = leverage
        self.fees = 0.0  # Trading fees
        self.slippage = 0.0  # Price slippage
        self.entry_reason = None  # Reason for entering the trade
        self.exit_reason = None  # Reason for exiting the trade
        self.market_conditions = {}  # Market conditions at entry
        self.technical_indicators = {}  # Technical indicators at entry
        self.trading_mode = trading_mode

    def update_pnl(self, current_price: float):
        """Update P&L for the trade with enhanced tracking."""
        try:
            # Calculate raw P&L
            if self.side == 'BUY':
                self.pnl = (current_price - self.entry_price) * self.quantity
            else:
                self.pnl = (self.entry_price - current_price) * self.quantity

            # Calculate P&L percentage
            self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100

            # Update max profit/loss
            if self.pnl > 0:
                self.max_profit = max(self.max_profit, self.pnl)
                # Check for trailing stop activation
                if self.pnl_percentage >= self.trailing_stop_activation * 100:
                    self._update_trailing_stop(current_price)
            else:
                self.max_loss = min(self.max_loss, self.pnl)

            # Calculate risk metrics
            self._calculate_risk_metrics(current_price)

        except Exception as e:
            logging.error(f"Error updating P&L: {e}")

    def _update_trailing_stop(self, current_price: float):
        """Update trailing stop level."""
        if self.side == 'BUY':
            new_stop = current_price * (1 - self.trailing_stop_distance)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:
            new_stop = current_price * (1 + self.trailing_stop_distance)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop

    def _calculate_risk_metrics(self, current_price: float):
        """Calculate risk metrics for the trade."""
        try:
            # Calculate position size
            self.position_size = self.quantity * current_price

            # Calculate risk amount
            if self.side == 'BUY':
                self.risk_amount = (self.entry_price - self.stop_loss) * self.quantity if self.stop_loss else 0
            else:
                self.risk_amount = (self.stop_loss - self.entry_price) * self.quantity if self.stop_loss else 0

            # Calculate risk-adjusted return
            if self.risk_amount > 0:
                self.risk_reward_ratio = abs(self.pnl / self.risk_amount)

        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")

    def close(self, exit_price: float, reason: str = None):
        """Close the trade with enhanced tracking."""
        try:
            self.exit_price = exit_price
            self.exit_time = datetime.now()
            self.status = 'CLOSED'
            self.exit_reason = reason
            self.update_pnl(exit_price)

            # Calculate final metrics
            self._calculate_final_metrics()

        except Exception as e:
            logging.error(f"Error closing trade: {e}")

    def _calculate_final_metrics(self):
        """Calculate final trade metrics."""
        try:
            # Calculate holding period
            holding_period = (self.exit_time - self.entry_time).total_seconds() / 3600  # in hours

            # Calculate return metrics
            total_return = self.pnl_percentage
            annualized_return = (total_return / holding_period) * 24 * 365 if holding_period > 0 else 0

            # Store final metrics
            self.final_metrics = {
                'holding_period': holding_period,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'risk_reward_ratio': self.risk_reward_ratio,
                'max_profit': self.max_profit,
                'max_loss': self.max_loss,
                'fees': self.fees,
                'slippage': self.slippage
            }

        except Exception as e:
            logging.error(f"Error calculating final metrics: {e}")

class TradeManager:
    def __init__(self, max_trade_duration: int = 24):
        self.max_trade_duration = max_trade_duration
        self.trades = {}
        self.trade_history = []
        self.daily_stats = {
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_profit': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_holding_time': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0
        }
        self.last_stats_reset = datetime.now().date()
        self.risk_limits = {
            'max_daily_trades': 5,
            'max_open_trades': 2,
            'max_daily_loss': 20.0,
            'max_position_size': 50.0,
            'max_leverage': 10,
            'min_profit_target': 0.005,
            'max_loss_per_trade': 0.01
        }
        self.performance_metrics = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

    def get_open_trades(self) -> Dict[str, Trade]:
        """Get all open trades."""
        return self.trades

    def remove_trade(self, order_id: str):
        """Remove a trade from the manager."""
        try:
            if order_id in self.trades:
                trade = self.trades[order_id]
                self.trade_history.append(trade)
                del self.trades[order_id]
                self.update_trade_stats(trade)
                logging.info(f"Removed trade {order_id} from manager")
        except Exception as e:
            logging.error(f"Error removing trade: {e}")

    def check_timeouts(self) -> List[Trade]:
        """Check for trades that have exceeded their maximum duration."""
        try:
            timed_out_trades = []
            current_time = datetime.now()
            
            for trade in list(self.trades.values()):
                duration = (current_time - trade.entry_time).total_seconds() / 3600
                if duration >= self.max_trade_duration:
                    timed_out_trades.append(trade)
                    logging.warning(f"Trade {trade.order_id} timed out after {duration:.2f} hours")
            
            return timed_out_trades
        except Exception as e:
            logging.error(f"Error checking timeouts: {e}")
            return []

    def add_trade(self, trade: Trade):
        """Add a new trade with enhanced validation."""
        try:
            # Check risk limits
            if not self._check_risk_limits(trade):
                logging.warning(f"Trade rejected due to risk limits: {trade.symbol}")
                return False

            # Add trade
            self.trades[trade.order_id] = trade
            logging.info(f"Added new trade: {trade.symbol} {trade.side} at {trade.entry_price}")
            return True

        except Exception as e:
            logging.error(f"Error adding trade: {e}")
            return False

    def _check_risk_limits(self, trade: Trade) -> bool:
        """Check if trade meets risk management criteria."""
        try:
            # Check number of open trades
            if len(self.trades) >= self.risk_limits['max_open_trades']:
                return False

            # Check daily trade limit
            if self.daily_stats['trades'] >= self.risk_limits['max_daily_trades']:
                return False

            # Check position size
            if trade.position_size > self.risk_limits['max_position_size']:
                return False

            # Check leverage
            if trade.leverage > self.risk_limits['max_leverage']:
                return False

            # Check daily loss limit
            if self.daily_stats['total_loss'] >= self.risk_limits['max_daily_loss']:
                return False

            return True

        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return False

    def update_trade_stats(self, trade: Trade):
        """Update trading statistics with enhanced metrics."""
        try:
            if trade.status == 'CLOSED':
                # Update basic stats
                self.daily_stats['trades'] += 1
                if trade.pnl > 0:
                    self.daily_stats['winning_trades'] += 1
                    self.daily_stats['total_profit'] += trade.pnl
                    self.daily_stats['largest_win'] = max(self.daily_stats['largest_win'], trade.pnl)
                else:
                    self.daily_stats['losing_trades'] += 1
                    self.daily_stats['total_loss'] += abs(trade.pnl)
                    self.daily_stats['largest_loss'] = min(self.daily_stats['largest_loss'], trade.pnl)

                # Update advanced metrics
                self._update_advanced_metrics(trade)

        except Exception as e:
            logging.error(f"Error updating trade stats: {e}")

    def _update_advanced_metrics(self, trade: Trade):
        """Update advanced trading metrics."""
        try:
            # Calculate win rate
            total_trades = self.daily_stats['winning_trades'] + self.daily_stats['losing_trades']
            if total_trades > 0:
                self.daily_stats['win_rate'] = self.daily_stats['winning_trades'] / total_trades

            # Calculate profit factor
            if self.daily_stats['total_loss'] > 0:
                self.daily_stats['profit_factor'] = self.daily_stats['total_profit'] / self.daily_stats['total_loss']

            # Calculate average profit/loss
            if self.daily_stats['winning_trades'] > 0:
                self.daily_stats['average_profit'] = self.daily_stats['total_profit'] / self.daily_stats['winning_trades']
            if self.daily_stats['losing_trades'] > 0:
                self.daily_stats['average_loss'] = self.daily_stats['total_loss'] / self.daily_stats['losing_trades']

            # Update fees and slippage
            self.daily_stats['total_fees'] += trade.fees
            self.daily_stats['total_slippage'] += trade.slippage

            # Calculate drawdown
            current_drawdown = (self.daily_stats['total_loss'] / 
                              (self.daily_stats['total_profit'] + self.daily_stats['total_loss']))
            self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], current_drawdown)

        except Exception as e:
            logging.error(f"Error updating advanced metrics: {e}")

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        try:
            return {
                'daily_stats': self.daily_stats,
                'risk_metrics': {
                    'max_drawdown': self.daily_stats['max_drawdown'],
                    'win_rate': self.daily_stats['win_rate'],
                    'profit_factor': self.daily_stats['profit_factor'],
                    'risk_reward_ratio': self._calculate_average_risk_reward()
                },
                'trade_metrics': {
                    'total_trades': self.daily_stats['trades'],
                    'winning_trades': self.daily_stats['winning_trades'],
                    'losing_trades': self.daily_stats['losing_trades'],
                    'average_profit': self.daily_stats['average_profit'],
                    'average_loss': self.daily_stats['average_loss']
                },
                'financial_metrics': {
                    'total_profit': self.daily_stats['total_profit'],
                    'total_loss': self.daily_stats['total_loss'],
                    'net_profit': self.daily_stats['total_profit'] - self.daily_stats['total_loss'],
                    'total_fees': self.daily_stats['total_fees'],
                    'total_slippage': self.daily_stats['total_slippage']
                }
            }
        except Exception as e:
            logging.error(f"Error generating performance report: {e}")
            return {}

    def _calculate_average_risk_reward(self) -> float:
        """Calculate average risk-reward ratio."""
        try:
            risk_reward_ratios = [trade.risk_reward_ratio for trade in self.trade_history]
            return sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0
        except Exception as e:
            logging.error(f"Error calculating average risk-reward: {e}")
            return 0.0

    def reset_daily_stats(self):
        """Reset daily statistics at the start of a new day."""
        current_date = datetime.now().date()
        if current_date > self.last_stats_reset:
            self.daily_stats = {
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_profit': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'average_holding_time': 0.0,
                'total_fees': 0.0,
                'total_slippage': 0.0
            }
            self.last_stats_reset = current_date
            logging.info("Daily statistics have been reset")

    def execute_futures_trade(self, symbol: str, side: str, quantity: float, leverage: int = 10) -> Optional[Trade]:
        """Execute a futures trade."""
        try:
            # التأكد من أن الرافعة المالية لا تتجاوز 10x
            leverage = min(leverage, 10)
            self.set_leverage(symbol, leverage)
            logging.info(f"Executing {side} futures order for {symbol}")
            logging.info(f"Quantity: {quantity}")
            logging.info(f"Leverage: {leverage}x")
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Create trade object
            trade = Trade(
                order_id=order['orderId'],
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=float(order['avgFillPrice']) if 'avgFillPrice' in order else None,
                trading_mode='futures',
                leverage=leverage
            )
            
            logging.info(f"Futures order executed successfully")
            logging.info(f"Order ID: {order['orderId']}")
            logging.info(f"Fill price: {trade.entry_price}")
            return trade
            
        except Exception as e:
            logging.error(f"Error executing futures trade: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int = 10):
        """Set leverage for a symbol on Binance Futures."""
        try:
            # التأكد من أن الرافعة المالية لا تتجاوز 10x
            leverage = min(leverage, 10)
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logging.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logging.error(f"Error setting leverage for {symbol}: {e}")

class RiskManager:
    def __init__(self):
        self.max_balance = 0.0
        self.initial_balance = 0.0
        self.daily_profit = 0.0
        self.emergency_stop = False
        self.risk_parameters = {
            'max_daily_loss_percentage': 0.05,
            'max_drawdown_percentage': 0.10,
            'min_balance_threshold': 50.0,
            'max_position_percentage': 0.20,
            'profit_target_percentage': 0.02,
            'stop_loss_percentage': 0.01
        }

    def initialize(self, initial_balance: float):
        """Initialize risk manager with initial balance."""
        self.initial_balance = initial_balance
        self.max_balance = initial_balance

    def check_risk_limits(self, current_balance: float) -> bool:
        """Check if current trading state is within risk limits."""
        try:
            self.max_balance = max(self.max_balance, current_balance)
            
            if current_balance < self.risk_parameters['min_balance_threshold']:
                logging.warning(f"Balance below minimum threshold: {current_balance}")
                self.emergency_stop = True
                return False
            
            drawdown = (self.max_balance - current_balance) / self.max_balance
            if drawdown > self.risk_parameters['max_drawdown_percentage']:
                logging.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
                self.emergency_stop = True
                return False
            
            daily_loss_percentage = abs(min(0, self.daily_profit)) / self.initial_balance
            if daily_loss_percentage > self.risk_parameters['max_daily_loss_percentage']:
                logging.warning(f"Daily loss limit exceeded: {daily_loss_percentage:.2%}")
                self.emergency_stop = True
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return False

    def update_daily_profit(self, pnl: float):
        """Update daily profit/loss."""
        self.daily_profit += pnl

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_profit = 0.0

class MarketAnalyzer:
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.volatility_cache = {}
        self.volume_cache = {}

    def check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for trading."""
        try:
            # Get latest data
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                logging.warning(f"No data available for {symbol}")
                return False

            # Check volume
            avg_volume = df['volume'].rolling(window=24).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Log volume information
            logging.info(f"Volume analysis for {symbol}:")
            logging.info(f"Current volume: {current_volume:.2f}")
            logging.info(f"Average volume: {avg_volume:.2f}")
            logging.info(f"Volume ratio: {volume_ratio:.2f}")
            
            # Check if volume is sufficient
            if volume_ratio < 0.5:  # Volume is less than 50% of average
                logging.info(f"Low volume for {symbol} (ratio: {volume_ratio:.2f}), skipping trade")
                return False
            
            # Check volatility
            volatility = self.calculate_volatility(symbol)
            if volatility > 0.05:  # High volatility threshold
                logging.info(f"High volatility for {symbol} ({volatility:.4f}), skipping trade")
                return False
            
            # Check price movement
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            if abs(price_change) > 5:  # More than 5% price change
                logging.info(f"Large price movement for {symbol} ({price_change:.2f}%), skipping trade")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market conditions for {symbol}: {e}")
            return False

    def _check_volume(self, symbol: str) -> bool:
        """Check if trading volume is sufficient."""
        try:
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                return False
                
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            
            if current_volume < avg_volume * 0.5:  # Low volume
                logging.info(f"Low volume for {symbol}, skipping trade")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking volume: {e}")
            return False

    def _check_volatility(self, symbol: str) -> bool:
        """Check if market volatility is within acceptable range."""
        try:
            if symbol in self.volatility_cache:
                cache_time, volatility = self.volatility_cache[symbol]
                if (datetime.now() - cache_time).seconds < 300:  # Cache for 5 minutes
                    return volatility <= 0.05  # High volatility threshold
            
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                return False
                
            returns = df['close'].pct_change().dropna()
            if len(returns) == 0:
                return False
                
            volatility = returns.std() * np.sqrt(24)  # 24-hour volatility
            self.volatility_cache[symbol] = (datetime.now(), volatility)
            
            if volatility > 0.05:  # High volatility threshold
                logging.info(f"High volatility for {symbol}, skipping trade")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking volatility: {e}")
            return False

    def _check_market_depth(self, symbol: str) -> bool:
        """Check if market depth is sufficient."""
        try:
            depth = self.data_collector.get_market_depth(symbol)
            
            # Check spread
            if depth['spread_percentage'] > 0.5:  # 0.5% spread threshold
                logging.info(f"High spread for {symbol}, skipping trade")
                return False
            
            # Check depth ratio
            if depth['depth_ratio'] < 0.5 or depth['depth_ratio'] > 2.0:
                logging.info(f"Unbalanced market depth for {symbol}, skipping trade")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market depth: {e}")
            return False

class TradingBot:
    def __init__(self):
        """Initialize the trading bot."""
        try:
            # Validate configuration
            validate_config()
            
            # Initialize client with API keys from config
            logging.info("Connecting to Binance API...")
            self.client = Client(API_KEY, API_SECRET)
            
            # Test connection
            try:
                self.client.ping()
                logging.info("Successfully connected to Binance API")
            except Exception as e:
                logging.error(f"Failed to connect to Binance API: {e}")
                raise Exception("Could not connect to Binance API. Please check your internet connection and API keys.")
            
            # Create config dictionary
            self.config = {
                'trading_pairs': TRADING_PAIRS,
                'prediction_timeframe': PREDICTION_TIMEFRAME,
                'lookback_period': LOOKBACK_PERIOD,
                'trading_mode': 'spot'  # Default to spot trading
            }
            
            logging.info("Initializing components...")
            self.data_collector = DataCollector(self.client, self.config)
            self.models = {}
            self.trade_manager = TradeManager()
            self.risk_manager = RiskManager()
            self.market_analyzer = MarketAnalyzer(self.data_collector)
            self.emergency_stop = False
            self.last_trade_time = {}
            self.volatility_cache = {}
            self.daily_profit = 0.0
            
            # Initialize balances for both spot and futures
            self.spot_balance = self.get_spot_balance()
            self.futures_balance = self.get_futures_balance()
            self.initial_balance = self.spot_balance + self.futures_balance
            self.max_balance = self.initial_balance
            
            self.trade_history = []
            self.daily_stats = {
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'max_drawdown': 0.0
            }
            self.last_stats_reset = datetime.now().date()
            
            logging.info("Bot initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Error initializing trading bot: {e}")
            raise

    def get_spot_balance(self) -> float:
        """Get current spot account balance in USDT."""
        try:
            account = self.client.get_account()
            balance = float([asset for asset in account['balances'] if asset['asset'] == 'USDT'][0]['free'])
            logging.info(f"Current spot balance: {balance} USDT")
            return balance
        except Exception as e:
            logging.error(f"Error getting spot balance: {e}")
            return 0.0

    def get_futures_balance(self) -> float:
        """Get current futures account balance in USDT."""
        try:
            account = self.client.futures_account_balance()
            balance = float([asset for asset in account if asset['asset'] == 'USDT'][0]['balance'])
            logging.info(f"Current futures balance: {balance} USDT")
            return balance
        except Exception as e:
            logging.error(f"Error getting futures balance: {e}")
            return 0.0

    def execute_spot_trade(self, symbol: str, side: str, quantity: float) -> Optional[Trade]:
        """Execute a spot trade."""
        try:
            logging.info(f"Executing {side} spot order for {symbol}")
            logging.info(f"Quantity: {quantity}")
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Create trade object
            trade = Trade(
                order_id=order['orderId'],
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=float(order['fills'][0]['price']) if 'fills' in order else None,
                trading_mode='spot'
            )
            
            logging.info(f"Spot order executed successfully")
            logging.info(f"Order ID: {order['orderId']}")
            logging.info(f"Fill price: {trade.entry_price}")
            return trade
            
        except Exception as e:
            logging.error(f"Error executing spot trade: {e}")
            return None

    def check_risk_limits(self) -> bool:
        """Check if current trading state is within risk limits."""
        spot_balance = self.get_spot_balance()
        futures_balance = self.get_futures_balance()
        total_balance = spot_balance + futures_balance
        
        # Update max balance
        self.max_balance = max(self.max_balance, total_balance)
        
        # Check drawdown
        drawdown = (self.max_balance - total_balance) / self.max_balance
        if drawdown > MAX_DRAWDOWN_PERCENTAGE:
            logging.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            return False
        
        # Check daily loss
        if self.daily_profit < -DAILY_LOSS_LIMIT:
            logging.warning(f"Daily loss limit exceeded: {self.daily_profit}")
            return False
        
        # Check minimum balance for each mode
        if self.config['trading_mode'] == 'spot' and spot_balance < 10.0:
            logging.warning(f"Spot balance below minimum threshold: {spot_balance}")
            return False
        elif self.config['trading_mode'] == 'futures' and futures_balance < 10.0:
            logging.warning(f"Futures balance below minimum threshold: {futures_balance}")
            return False
        
        return True

    def calculate_volatility(self, symbol: str) -> float:
        """Calculate current market volatility."""
        if symbol in self.volatility_cache:
            cache_time, volatility = self.volatility_cache[symbol]
            if (datetime.now() - cache_time).seconds < 300:  # Cache for 5 minutes
                return volatility
        
        try:
            df = self.data_collector.get_latest_data(symbol)
            if df.empty or len(df) < 2:
                return 0.0
            
            returns = df['close'].pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            volatility = returns.std() * np.sqrt(24)  # 24-hour volatility
            self.volatility_cache[symbol] = (datetime.now(), volatility)
            return volatility
        except Exception as e:
            logging.error(f"Error calculating volatility: {e}")
            return 0.0

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            account = self.client.get_account()
            balance = float([asset for asset in account['balances'] if asset['asset'] == 'USDT'][0]['free'])
            
            # Check for valid balance
            if balance <= 0:
                logging.warning("Insufficient balance for trading")
                return 0.0
            
            # Check for valid price
            if price <= 0:
                logging.warning("Invalid price value")
                return 0.0
            
            # Calculate position size based on risk management
            risk_percentage = 0.01  # 1% risk for each trade
            available_balance = balance * risk_percentage
            
            # Adjust position size based on volatility
            volatility = self.calculate_volatility(symbol)
            volatility_factor = max(0.3, 1 - volatility)  # Reduce position size in high volatility
            
            # Calculate position size based on risk management
            position_size = (available_balance / price) * volatility_factor
            
            # Ensure position size doesn't exceed available balance
            max_position = balance * 0.2  # 20% of balance as max position size
            return min(position_size, max_position)
        
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.0

    def update_trade_stats(self, trade: Trade):
        """Update trading statistics."""
        if trade.status == 'CLOSED':
            self.daily_stats['trades'] += 1
            if trade.pnl > 0:
                self.daily_stats['winning_trades'] += 1
                self.daily_stats['total_profit'] += trade.pnl
            else:
                self.daily_stats['losing_trades'] += 1
                self.daily_stats['total_loss'] += abs(trade.pnl)
            
            # Update max drawdown
            current_drawdown = (self.daily_stats['total_loss'] / 
                              (self.daily_stats['total_profit'] + self.daily_stats['total_loss']))
            self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], current_drawdown)

    def reset_daily_stats(self):
        """Reset daily statistics at the start of a new day."""
        current_date = datetime.now().date()
        if current_date > self.last_stats_reset:
            self.daily_stats = {
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'max_drawdown': 0.0
            }
            self.last_stats_reset = current_date

    def save_trade_history(self):
        """Save trade history to file."""
        try:
            trade_data = []
            for trade in self.trade_history:
                trade_data.append({
                    'order_id': trade.order_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'entry_price': trade.entry_price,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_price': trade.exit_price,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'status': trade.status,
                    'pnl': trade.pnl,
                    'pnl_percentage': trade.pnl_percentage,
                    'max_profit': trade.max_profit,
                    'max_loss': trade.max_loss
                })
            
            with open('trade_history.json', 'w') as f:
                json.dump(trade_data, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error saving trade history: {e}")

    def check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for trading."""
        try:
            # Get latest data
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                logging.warning(f"No data available for {symbol}")
                return False

            # Check volume
            avg_volume = df['volume'].rolling(window=24).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Log volume information
            logging.info(f"Volume analysis for {symbol}:")
            logging.info(f"Current volume: {current_volume:.2f}")
            logging.info(f"Average volume: {avg_volume:.2f}")
            logging.info(f"Volume ratio: {volume_ratio:.2f}")
            
            # Check if volume is sufficient
            if volume_ratio < 0.5:  # Volume is less than 50% of average
                logging.info(f"Low volume for {symbol} (ratio: {volume_ratio:.2f}), skipping trade")
                return False
            
            # Check volatility
            volatility = self.calculate_volatility(symbol)
            if volatility > 0.05:  # High volatility threshold
                logging.info(f"High volatility for {symbol} ({volatility:.4f}), skipping trade")
                return False
            
            # Check price movement
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            if abs(price_change) > 5:  # More than 5% price change
                logging.info(f"Large price movement for {symbol} ({price_change:.2f}%), skipping trade")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market conditions for {symbol}: {e}")
            return False

    def emergency_stop_trading(self):
        """Emergency stop all trading activities."""
        try:
            self.emergency_stop = True
            logging.warning("EMERGENCY STOP ACTIVATED - Closing all positions")
            
            open_trades = self.trade_manager.get_open_trades()
            if not open_trades:
                logging.info("No open trades to close")
                return
            
            for order_id, trade in list(open_trades.items()):
                try:
                    close_side = 'SELL' if trade.side == 'BUY' else 'BUY'
                    order = self.execute_futures_trade(trade.symbol, close_side, trade.quantity)
                    if order:
                        trade.close(float(order['avgFillPrice']), "EMERGENCY_STOP")
                        self.trade_manager.remove_trade(order_id)
                        logging.info(f"Closed position for {trade.symbol} during emergency stop")
                except Exception as e:
                    logging.error(f"Error closing position during emergency stop: {e}")
                
        except Exception as e:
            logging.error(f"Error in emergency stop: {e}")

    def initialize_models(self):
        """تهيئة النماذج للتحليل"""
        try:
            logging.info("Starting model initialization...")
            
            for symbol in TRADING_PAIRS:
                logging.info(f"Initializing model for {symbol}...")
                
                # الحصول على تحليل TradingView
                handler = TA_Handler(
                    symbol=symbol,
                    screener="crypto",
                    exchange="BINANCE",
                    interval=Interval.INTERVAL_1_HOUR
                )
                
                try:
                    analysis = handler.get_analysis()
                    if analysis:
                        logging.info(f"Successfully initialized model for {symbol}")
                        self.models[symbol] = analysis
                    else:
                        logging.error(f"Failed to get analysis for {symbol}")
                except Exception as e:
                    logging.error(f"Error initializing model for {symbol}: {e}")
                    continue

            if not self.models:
                raise Exception("Failed to initialize any models")
            
            logging.info("Model initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Error in initialize_models: {e}")
            raise Exception("Failed to initialize any models")

    def get_historical_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Get historical klines with enhanced error handling."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Fetching historical data for {symbol} (attempt {attempt + 1}/{max_retries})...")
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                if not klines:
                    raise Exception(f"No data received for {symbol}")
                    
                logging.info(f"Successfully fetched {len(klines)} data points for {symbol}")
                return klines
                
            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp for this request is outside of the recvWindow
                    logging.warning("Timestamp issue, retrying with adjusted time...")
                    time_module.sleep(1)
                elif e.code == -1015:  # Too many requests
                    logging.warning("Rate limit exceeded, waiting before retry...")
                    time_module.sleep(retry_delay * (attempt + 1))
                else:
                    logging.error(f"Binance API error: {e.code} - {e.message}")
                    if attempt < max_retries - 1:
                        time_module.sleep(retry_delay)
                    else:
                        raise
                    
            except Exception as e:
                logging.error(f"Error fetching historical data: {e}")
                if attempt < max_retries - 1:
                    time_module.sleep(retry_delay)
                else:
                    raise
        
        raise Exception(f"Failed to fetch historical data for {symbol} after {max_retries} attempts")

    def is_trading_time(self):
        """Check if current time is within trading hours."""
        current_time = datetime.now().time()
        return TRADING_START_TIME <= current_time <= TRADING_END_TIME
    
    def check_stop_loss_take_profit(self, symbol: str, entry_price: float, current_price: float):
        """Check if stop loss or take profit levels have been hit."""
        if symbol not in self.trade_manager.get_open_trades():
            return False
        
        trade = self.trade_manager.get_open_trades()[symbol]
        if trade['side'] == 'BUY':
            # Long position
            if current_price <= entry_price * (1 - STOP_LOSS_PERCENTAGE):
                return 'STOP_LOSS'
            if current_price >= entry_price * (1 + TAKE_PROFIT_PERCENTAGE):
                return 'TAKE_PROFIT'
        else:
            # Short position
            if current_price >= entry_price * (1 + STOP_LOSS_PERCENTAGE):
                return 'STOP_LOSS'
            if current_price <= entry_price * (1 - TAKE_PROFIT_PERCENTAGE):
                return 'TAKE_PROFIT'
        
        return False
    
    def get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and not expired."""
        try:
            if symbol in self.data_cache:
                last_update = self.last_cache_update.get(symbol, 0)
                current_time = time_module.time()
                
                if current_time - last_update < self.cache_timeout:
                    return self.data_cache[symbol]
                    
            return None
            
        except Exception as e:
            logging.error(f"Error getting cached data: {e}")
            return None

    def update_cache(self, symbol: str, data: pd.DataFrame):
        """Update data cache."""
        try:
            self.data_cache[symbol] = data
            self.last_cache_update[symbol] = time_module.time()
        except Exception as e:
            logging.error(f"Error updating cache: {e}")

    def run(self):
        """Main trading loop with enhanced performance."""
        logging.info("Starting trading bot...")
        logging.info("Initializing models...")
        self.initialize_models()
        
        logging.info("Starting main trading loop...")
        while not self.emergency_stop:
            try:
                current_time = time_module.time()
                
                # التحقق من وقت التداول
                if not self.is_trading_time():
                    logging.info("Outside trading hours. Waiting...")
                    time_module.sleep(60)
                    continue
                
                # إعادة تعيين الإحصائيات اليومية إذا لزم الأمر
                self.trade_manager.reset_daily_stats()
                
                # التحقق من الصفقات المنتهية
                timed_out_trades = self.trade_manager.check_timeouts()
                for trade in timed_out_trades:
                    self.close_trade(trade, "TIMEOUT")
                
                # التحقق من حدود المخاطر
                spot_balance = self.get_spot_balance()
                futures_balance = self.get_futures_balance()
                total_balance = spot_balance + futures_balance
                
                if not self.risk_manager.check_risk_limits(total_balance):
                    logging.warning("Risk limits exceeded, pausing trading")
                    time_module.sleep(300)
                    continue
                
                # معالجة كل عملة
                for symbol in TRADING_PAIRS:
                    try:
                        # التحقق من وقت آخر معالجة
                        last_process = self.last_processing_time.get(symbol, 0)
                        if current_time - last_process < self.processing_interval:
                            continue
                        
                        logging.info(f"Processing {symbol}...")
                        
                        # التحقق من وقت آخر فحص للسوق
                        last_check = self.last_market_check.get(symbol, 0)
                        if current_time - last_check >= self.market_check_interval:
                            if not self.market_analyzer.check_market_conditions(symbol):
                                self.last_market_check[symbol] = current_time
                                continue
                            self.last_market_check[symbol] = current_time
                        
                        # الحصول على البيانات من الذاكرة المؤقتة أو من API
                        df = self.get_cached_data(symbol)
                        if df is None:
                            df = self.data_collector.get_latest_data(symbol)
                            if not df.empty:
                                self.update_cache(symbol, df)
                        
                        if df.empty:
                            continue
                        
                        current_price = float(df['close'].iloc[-1])
                        self.process_symbol(symbol, df, current_price)
                        
                        # تحديث وقت آخر معالجة
                        self.last_processing_time[symbol] = current_time
                        
                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {e}")
                        continue
                
                # انتظار قصيرة قبل الدورة التالية
                time_module.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time_module.sleep(60)
                continue

    def process_symbol(self, symbol: str, df: pd.DataFrame, current_price: float):
        """Process a single symbol with optimized performance."""
        try:
            model = self.models.get(symbol)
            if not model:
                return
                
            # تحضير البيانات للتنبؤ
            X = df[['open', 'high', 'low', 'close', 'volume',
                   'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14',
                   'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                   'ATRr_14', 'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0',
                   'OBV']].values[-24:]
            
            # التنبؤ
            prediction = model.predict(X)
            if prediction is None:
                return
                
            predicted_return = prediction[0][0]
            
            # التحقق من الصفقات المفتوحة
            open_trades = self.trade_manager.get_open_trades()
            for trade in list(open_trades.values()):
                if trade.symbol == symbol:
                    action = self.check_stop_loss_take_profit(
                        symbol, trade.entry_price, current_price
                    )
                    if action:
                        self.close_trade(trade, action)
                        continue
            
            # التحقق من فرص التداول الجديدة
            if len(open_trades) < MAX_OPEN_TRADES:
                if predicted_return > 0.001:
                    self.open_long_position(symbol, current_price)
                elif predicted_return < -0.001:
                    self.open_short_position(symbol, current_price)
                    
        except Exception as e:
            logging.error(f"Error processing symbol {symbol}: {e}")

    def open_long_position(self, symbol: str, current_price: float):
        """Open a long position."""
        try:
            # Check if we have enough balance
            if self.config['trading_mode'] == 'spot':
                balance = self.get_spot_balance()
            else:
                balance = self.get_futures_balance()
                
            if balance <= 0:
                logging.warning("Insufficient balance for trading")
                return

            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logging.warning(f"Invalid position size for {symbol}")
                return

            logging.info(f"=== Opening LONG position for {symbol} ===")
            logging.info(f"Current price: {current_price}")
            logging.info(f"Position size: {quantity}")
            logging.info(f"Account balance: {balance}")

            # Execute trade
            if self.config['trading_mode'] == 'spot':
                trade = self.execute_spot_trade(symbol, 'BUY', quantity)
            else:
                trade = self.execute_futures_trade(symbol, 'BUY', quantity)
                
            if trade:
                self.trade_manager.add_trade(trade)
                logging.info(f"=== LONG position opened successfully ===")
                logging.info(f"Entry price: {trade.entry_price}")
                logging.info(f"Stop loss: {trade.stop_loss}")
                logging.info(f"Take profit: {trade.take_profit}")
                logging.info(f"Order ID: {trade.order_id}")
                
        except Exception as e:
            logging.error(f"Error opening long position for {symbol}: {e}")

    def open_short_position(self, symbol: str, current_price: float):
        """Open a short position."""
        try:
            # Check if we have enough balance
            if self.config['trading_mode'] == 'spot':
                balance = self.get_spot_balance()
            else:
                balance = self.get_futures_balance()
                
            if balance <= 0:
                logging.warning("Insufficient balance for trading")
                return

            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logging.warning(f"Invalid position size for {symbol}")
                return

            logging.info(f"=== Opening SHORT position for {symbol} ===")
            logging.info(f"Current price: {current_price}")
            logging.info(f"Position size: {quantity}")
            logging.info(f"Account balance: {balance}")

            # Execute trade
            if self.config['trading_mode'] == 'spot':
                trade = self.execute_spot_trade(symbol, 'SELL', quantity)
            else:
                trade = self.execute_futures_trade(symbol, 'SELL', quantity)
                
            if trade:
                self.trade_manager.add_trade(trade)
                logging.info(f"=== SHORT position opened successfully ===")
                logging.info(f"Entry price: {trade.entry_price}")
                logging.info(f"Stop loss: {trade.stop_loss}")
                logging.info(f"Take profit: {trade.take_profit}")
                logging.info(f"Order ID: {trade.order_id}")
                
        except Exception as e:
            logging.error(f"Error opening short position for {symbol}: {e}")

    def close_trade(self, trade: Trade, reason: str):
        """Close a trade."""
        try:
            logging.info(f"=== Closing position for {trade.symbol} ===")
            logging.info(f"Reason: {reason}")
            logging.info(f"Entry price: {trade.entry_price}")
            logging.info(f"Current P&L: {trade.pnl:.2f} USDT ({trade.pnl_percentage:.2f}%)")
            
            close_side = 'SELL' if trade.side == 'BUY' else 'BUY'
            order = self.execute_futures_trade(trade.symbol, close_side, trade.quantity)
            if order:
                trade.close(float(order['avgFillPrice']), reason)
                self.trade_manager.remove_trade(trade.order_id)
                logging.info(f"=== Position closed successfully ===")
                logging.info(f"Exit price: {trade.exit_price}")
                logging.info(f"Final P&L: {trade.pnl:.2f} USDT ({trade.pnl_percentage:.2f}%)")
                logging.info(f"Trade duration: {(trade.exit_time - trade.entry_time).total_seconds() / 3600:.2f} hours")
                logging.info(f"Order ID: {order['orderId']}")
                
                # Update daily stats
                self.update_trade_stats(trade)
                logging.info(f"Daily stats updated - Total trades: {self.daily_stats['trades']}")
                logging.info(f"Winning trades: {self.daily_stats['winning_trades']}")
                logging.info(f"Total profit: {self.daily_stats['total_profit']:.2f} USDT")
                
        except Exception as e:
            logging.error(f"Error closing trade: {e}")

    def example_trading(self):
        """Example of how to use the trading functionality."""
        try:
            # Get current market data
            symbol = 'BTCUSDT'
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                logging.error("No data available for trading")
                return

            current_price = float(df['close'].iloc[-1])
            
            # Example 1: Opening a long position
            logging.info("=== Example 1: Opening a Long Position ===")
            self.open_long_position(symbol, current_price)
            
            # Example 2: Opening a short position
            logging.info("=== Example 2: Opening a Short Position ===")
            self.open_short_position(symbol, current_price)
            
            # Example 3: Checking open trades
            open_trades = self.trade_manager.get_open_trades()
            logging.info(f"Current open trades: {len(open_trades)}")
            
            # Example 4: Closing a trade
            if open_trades:
                trade = list(open_trades.values())[0]
                logging.info("=== Example 4: Closing a Trade ===")
                self.close_trade(trade, "MANUAL")
                
            # Example 5: Getting trading statistics
            logging.info("=== Example 5: Trading Statistics ===")
            logging.info(f"Daily trades: {self.daily_stats['trades']}")
            logging.info(f"Winning trades: {self.daily_stats['winning_trades']}")
            logging.info(f"Total profit: {self.daily_stats['total_profit']:.2f} USDT")
            
        except Exception as e:
            logging.error(f"Error in example trading: {e}")

    def manual_trade(self, symbol: str, side: str, quantity: float):
        """Execute a manual trade."""
        try:
            # Get current price
            df = self.data_collector.get_latest_data(symbol)
            if df.empty:
                logging.error("No data available for trading")
                return False
            
            current_price = float(df['close'].iloc[-1])
            
            # Execute trade
            if side.upper() == 'BUY':
                self.open_long_position(symbol, current_price)
            elif side.upper() == 'SELL':
                self.open_short_position(symbol, current_price)
            else:
                logging.error(f"Invalid side: {side}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in manual trade: {e}")
            return False

    def close_all_positions(self):
        """Close all open positions."""
        try:
            open_trades = self.trade_manager.get_open_trades()
            if not open_trades:
                logging.info("No open positions to close")
                return
            
            logging.info(f"Closing {len(open_trades)} positions...")
            for trade in list(open_trades.values()):
                self.close_trade(trade, "MANUAL")
            
            logging.info("All positions closed")
            
        except Exception as e:
            logging.error(f"Error closing all positions: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
        bot.emergency_stop_trading()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        bot.emergency_stop_trading() 