from datetime import datetime, time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Trading Parameters
TRADING_PAIRS = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'SOLUSDT',
    'XRPUSDT',
    'ADAUSDT',
    'AVAXUSDT',
    'DOGEUSDT'
]

CAPITAL_PERCENTAGE = 0.05
DAILY_PROFIT_TARGET = 0.01
MAX_LOSS_PERCENTAGE = 0.01
MAX_DRAWDOWN_PERCENTAGE = 0.05
DAILY_LOSS_LIMIT = 50

# Trading Hours
TRADING_START_TIME = time(2, 0)  # 05:00 Saudi Time
TRADING_END_TIME = time(0, 0)    # 03:00 Saudi Time

# Model Parameters
PREDICTION_TIMEFRAME = '1h'
LOOKBACK_PERIOD = 24
TRAINING_EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Technical Indicators
INDICATORS = [
    'RSI',
    'MACD',
    'Bollinger Bands',
    'Moving Averages',
    'Volume Profile',
    'Market Structure',
    'Support/Resistance'
]

# Risk Management
STOP_LOSS_PERCENTAGE = 0.01
TAKE_PROFIT_PERCENTAGE = 0.02
TRAILING_STOP_PERCENTAGE = 0.005
MAX_OPEN_TRADES = 2
MAX_DAILY_LOSS = 0.02
POSITION_SIZING_METHOD = 'fixed'

# Market Monitoring
VOLUME_THRESHOLD = 2.0
VOLATILITY_THRESHOLD = 0.015
TREND_STRENGTH_THRESHOLD = 20
NEWS_IMPACT_THRESHOLD = 0.005

# Model Ensemble
USE_ENSEMBLE = True
ENSEMBLE_WEIGHTS = {
    'lstm': 0.4,
    'xgboost': 0.3,
    'lightgbm': 0.3
}

# Backtesting Parameters
BACKTEST_PERIODS = ['1M', '3M', '6M', '1Y']

# Error handling
MAX_RETRIES = 5
RETRY_DELAY = 3

# Market conditions
MIN_VOLUME_THRESHOLD = 0.8
MAX_VOLATILITY_THRESHOLD = 0.03

# Model parameters
TRAINING_DAYS = 180
PREDICTION_THRESHOLD = 0.002

# Trading parameters
TRADING_PARAMS = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    'timeframes': ['1h', '4h', '1d'],
    'base_currency': 'USDT',
    'max_open_trades': 2,
    'stake_amount': 50,
    'stake_currency': 'USDT',
    'dry_run': True,
}

# Risk management
RISK_PARAMS = {
    'max_risk_per_trade': 0.01,
    'max_daily_risk': 0.02,
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.02,
    'trailing_stop': True,
    'trailing_stop_positive': 0.005,
    'trailing_stop_positive_offset': 0.01,
    'trailing_only_offset_is_reached': True,
}

# Technical analysis parameters
TECHNICAL_PARAMS = {
    'lookback_period': 100,
    'min_data_points': 50,
    'indicators': {
        'sma': [20, 50, 200],
        'ema': [20, 50, 200],
        'rsi': 14,
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger': {'length': 20, 'std': 2},
        'atr': 14,
    },
    'signal_threshold': 0.7,
}

# Trading hours
TRADING_HOURS = {
    'start_hour': 0,
    'end_hour': 24,
    'exclude_weekends': True,
    'exclude_holidays': True,
}

# API configuration
API_CONFIG = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY'),
        'api_secret': os.getenv('BINANCE_API_SECRET'),
        'testnet': True,
        'recv_window': 5000,
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
    'max_size': 10 * 1024 * 1024,
    'backup_count': 5,
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'track_trades': True,
    'save_trade_history': True,
    'calculate_metrics': True,
    'metrics': ['sharpe_ratio', 'sortino_ratio', 'max_drawdown'],
    'backtest': {
        'enabled': True,
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
    }
}

# Error handling
ERROR_HANDLING = {
    'max_retries': 5,
    'retry_delay': 2,
    'exponential_backoff': True,
    'notify_on_error': True,
}

# Database configuration
DATABASE_CONFIG = {
    'enabled': True,
    'type': 'sqlite',
    'path': 'trading_bot.db',
    'backup': True,
    'backup_interval': 'daily',
}

# Notification settings
NOTIFICATION_CONFIG = {
    'enabled': True,
    'methods': ['email', 'telegram'],
    'notify_on_trade': True,
    'notify_on_error': True,
    'notify_on_profit': True,
    'profit_threshold': 0.02,
}

def validate_config():
    if not API_KEY or not API_SECRET:
        raise ValueError("Binance API keys not found in .env file")
    return True 