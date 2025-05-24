import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import ta
import pandas_ta as pta
from config import TRADING_PAIRS, PREDICTION_TIMEFRAME, LOOKBACK_PERIOD
from binance.exceptions import BinanceAPIException
import logging
from typing import Dict, List, Optional, Tuple
import time as time_module
from technical_analysis import TechnicalAnalysis
import pytz

class DataCollector:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.timezone = pytz.UTC
        self.technical_analysis = TechnicalAnalysis()

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """Get historical klines and convert to DataFrame with proper formatting."""
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.now(self.timezone)
            if start_time is None:
                start_time = end_time - timedelta(days=30)

            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            # Get klines with retry mechanism
            self.logger.info(f"Fetching klines for {symbol} via _get_klines_with_retry...")
            klines = self._get_klines_with_retry(
                symbol=symbol,
                interval=interval,
                start_time=start_ts,
                end_time=end_ts,
                limit=limit
            )

            if not klines:
                self.logger.error(f"No data received for {symbol}")
                return pd.DataFrame()

            self.logger.info(f"Successfully received {len(klines)} klines for {symbol}")
            if klines:
                self.logger.info(f"Sample kline structure: {klines[0]}")
                self.logger.info(f"Sample kline types: {[(k, type(v)) for k, v in klines[0].items()]}")

            # Create DataFrame with explicit column names
            column_names = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            self.logger.info(f"Creating DataFrame for {symbol} with columns: {column_names}")
            df = pd.DataFrame(klines)
            
            self.logger.info(f"DF_LOG_STEP_2A - Columns after creation: {df.columns.tolist()}")
            self.logger.info(f"DF_LOG_STEP_2B - Dtypes after creation:\n{df.dtypes}")

            # Convert timestamp to datetime and set as index
            if 'open_time' not in df.columns:
                self.logger.error(f"'open_time' column missing. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            self.logger.info(f"DF_LOG_STEP_3 - After setting index. Columns: {df.columns.tolist()}")

            # Convert OHLCV columns to numeric
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if col not in df.columns:
                    self.logger.error(f"Column '{col}' missing. Available columns: {df.columns.tolist()}")
                    return pd.DataFrame()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                self.logger.info(f"Converted {col} to numeric. Sample value: {df[col].iloc[0] if not df.empty else 'N/A'}")

            self.logger.info(f"DF_LOG_STEP_4 - Dtypes after numeric conversion:\n{df.dtypes}")

            # Keep only essential OHLCV columns
            try:
                df = df[ohlcv_columns].copy()
                self.logger.info(f"DF_LOG_STEP_5 - Selected OHLCV columns. Shape: {df.shape}")
            except KeyError as e:
                self.logger.error(f"KeyError during column selection: {e}. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()

            # Validate data quality
            df = self._validate_data_quality(df)
            if df.empty:
                self.logger.error(f"No valid data after quality validation for {symbol}")
                return pd.DataFrame()

            self.logger.info(f"DF_LOG_STEP_6 - After data validation. Shape: {df.shape}")
            self.logger.info(f"DF_LOG_STEP_6 - Dtypes:\n{df.dtypes}")

            # Add technical indicators
            self.logger.info(f"DF_LOG_STEP_7_PRE_TA - Before adding indicators. Columns: {df.columns.tolist()}")
            df = self.add_technical_indicators(df)
            
            if df.empty:
                self.logger.error(f"No data after adding indicators for {symbol}")
                return pd.DataFrame()

            self.logger.info(f"Successfully processed data for {symbol}. Final shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error in get_historical_klines for {symbol}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _get_klines_with_retry(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int,
        max_retries: int = 5
    ) -> List[Dict]:
        """Get klines with improved retry mechanism and timestamp handling."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time_module.sleep(2 ** attempt)  # Exponential backoff

                # Fetch klines
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=limit
                )

                if not klines:
                    self.logger.warning(f"No klines returned for {symbol} on attempt {attempt + 1}")
                    continue

                # Process klines into list of dictionaries with proper types
                processed_klines = []
                for kline in klines:
                    try:
                        processed_kline = {
                            'open_time': int(kline[0]),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'close_time': int(kline[6]),
                            'quote_asset_volume': float(kline[7]),
                            'number_of_trades': int(kline[8]),
                            'taker_buy_base_asset_volume': float(kline[9]),
                            'taker_buy_quote_asset_volume': float(kline[10]),
                            'ignore': kline[11]
                        }
                        processed_klines.append(processed_kline)
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error processing kline: {e}. Kline data: {kline}")
                        continue

                if not processed_klines:
                    self.logger.error(f"No valid klines processed for {symbol}")
                    continue

                # Log sample of processed data
                self.logger.info(f"Successfully processed {len(processed_klines)} klines for {symbol}")
                self.logger.info(f"Sample processed kline: {processed_klines[0]}")
                self.logger.info(f"Sample kline types: {[(k, type(v)) for k, v in processed_klines[0].items()]}")

                return processed_klines

            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp error
                    self.logger.warning(f"Timestamp error on attempt {attempt + 1}, retrying...")
                    continue
                if e.code == -1000:  # Rate limit
                    self.logger.warning(f"Rate limit hit on attempt {attempt + 1}, waiting...")
                    time_module.sleep(5)
                    continue
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                continue

        return []

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data with improved quality checks."""
        if df.empty:
            return df

        try:
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                self.logger.warning(f"Missing values found: {missing_values}")
                df = df.ffill().bfill()  # Forward fill then backward fill

            # Remove rows with invalid values (non-positive or NaN)
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in critical_columns:
                invalid_mask = (df[col] <= 0) | df[col].isnull()
                if invalid_mask.any():
                    self.logger.warning(f"Found {invalid_mask.sum()} invalid values in {col}")
                    df = df[~invalid_mask].copy() # Use .copy() to avoid potential warnings

            if df.empty:
                self.logger.warning("DataFrame is empty after removing invalid values.")
                return df

            # Check for price anomalies (large percentage changes)
            df['price_change'] = df['close'].pct_change()
            price_anomalies = (abs(df['price_change']) > 0.5)  # 50% price change
            if price_anomalies.any():
                self.logger.warning(f"Found {price_anomalies.sum()} price anomalies")
                df = df[~price_anomalies].copy()

            # Check for volume anomalies (significant deviation from moving average)
            if not df.empty and 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
                volume_anomalies = (df['volume'] > df['volume_ma'] * 5) & (df['volume_ma'] > 0)
                if volume_anomalies.any():
                    self.logger.warning(f"Found {volume_anomalies.sum()} volume anomalies")
                    # Cap volume instead of removing
                    df.loc[volume_anomalies, 'volume'] = df.loc[volume_anomalies, 'volume_ma'] * 5

            # Check for data gaps and fill them
            expected_freq = self._get_expected_frequency(df.index)
            if expected_freq:
                original_length = len(df)
                df = df.asfreq(expected_freq)
                if len(df) > original_length:
                    gaps = df.isnull().sum().sum()
                    self.logger.warning(f"Filled {gaps} data points due to gaps with frequency: {expected_freq}")
                    df = df.ffill() # Fill forward after reindexing

            return df

        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            return df

    def _get_expected_frequency(self, index: pd.DatetimeIndex) -> Optional[str]:
        """Get expected frequency based on data."""
        if len(index) < 2:
            return None

        # Calculate time differences
        time_diffs = index.to_series().diff().dropna()

        if time_diffs.empty:
            return None

        # Get most common time difference
        most_common = time_diffs.mode()
        if not most_common.empty:
            most_common = most_common.iloc[0]
        else:
            return None

        # Map to pandas frequency
        if most_common == pd.Timedelta(minutes=1):
            return '1min'
        elif most_common == pd.Timedelta(minutes=5):
            return '5min'
        elif most_common == pd.Timedelta(minutes=15):
            return '15min'
        elif most_common == pd.Timedelta(minutes=30):
            return '30min'
        elif most_common == pd.Timedelta(hours=1):
            return '1h'
        elif most_common == pd.Timedelta(hours=4):
            return '4h'
        elif most_common == pd.Timedelta(days=1):
            return '1d'

        return None

    def get_market_data(
        self,
        symbol: str,
        interval: str,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """Get market data with proper error handling and validation."""
        try:
            # Calculate time range
            end_time = datetime.now(self.timezone)
            start_time = end_time - timedelta(days=lookback_days)

            # Get historical klines
            df = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )

            if df.empty:
                self.logger.error(f"No data available for {symbol} {interval}")
                return pd.DataFrame()

            # Add additional market data
            df = self._add_market_data(df, symbol)

            return df

        except Exception as e:
            self.logger.error(f"Error in get_market_data: {e}")
            return pd.DataFrame()

    def _add_market_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add additional market data to the DataFrame."""
        try:
            # Get 24h ticker
            ticker = self.client.get_ticker(symbol=symbol)

            # Add ticker data
            df['price_change_24h'] = float(ticker['priceChangePercent'])
            df['volume_24h'] = float(ticker['volume'])
            df['quote_volume_24h'] = float(ticker['quoteVolume'])

            # Get order book
            depth = self.client.get_order_book(symbol=symbol, limit=5)

            # Calculate spread
            best_bid = float(depth['bids'][0][0]) if depth['bids'] else np.nan
            best_ask = float(depth['asks'][0][0]) if depth['asks'] else np.nan
            spread = (best_ask - best_bid) / best_bid if best_bid != 0 and not np.isnan(best_bid) else np.nan

            # Add order book data
            df['spread'] = spread
            df['bid_price'] = best_bid
            df['ask_price'] = best_ask

            return df

        except Exception as e:
            self.logger.error(f"Error adding market data: {e}")
            return df

    def get_account_balance(self) -> Dict:
        """Get account balance with proper error handling."""
        try:
            account = self.client.get_account()
            balances = {
                asset['asset']: {
                    'free': float(asset['free']),
                    'locked': float(asset['locked'])
                }
                for asset in account['balances']
                if float(asset['free']) > 0 or float(asset['locked']) > 0
            }
            return balances

        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return {}

    def get_latest_data(self, symbol: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Get latest market data."""
        try:
            end_time = datetime.now(self.timezone)
            start_time = end_time - timedelta(hours=lookback_hours)

            return self.get_market_data(symbol, '1h', lookback_days=lookback_hours/24)

        except Exception as e:
            logging.error(f"Error getting latest data: {e}")
            return pd.DataFrame()

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get current order book."""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'bids': [[float(price), float(qty)] for price, qty in depth['bids']],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks']]
            }
        except Exception as e:
            logging.error(f"Error getting order book: {e}")
            return {'bids': [], 'asks': []}

    def get_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get recent trades."""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            df = pd.DataFrame(trades)

            # Convert types
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            for col in ['price', 'qty', 'quoteQty']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logging.error(f"Error getting recent trades: {e}")
            return pd.DataFrame()

    def get_market_depth(self, symbol: str) -> Dict[str, float]:
        """Calculate market depth metrics."""
        try:
            order_book = self.get_order_book(symbol)

            # Calculate bid-ask spread
            best_bid = float(order_book['bids'][0][0]) if order_book['bids'] else np.nan
            best_ask = float(order_book['asks'][0][0]) if order_book['asks'] else np.nan
            spread = (best_ask - best_bid) / best_bid if best_bid != 0 and not np.isnan(best_bid) else np.nan

            # Calculate depth
            bid_depth = sum(qty for _, qty in order_book['bids']) if order_book['bids'] else 0
            ask_depth = sum(qty for _, qty in order_book['asks']) if order_book['asks'] else 0

            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else float('inf') if bid_depth > 0 else np.nan

            return {
                'spread': spread,
                'spread_percentage': (spread / best_bid) * 100 if not np.isnan(best_bid) and best_bid != 0 else np.nan,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'depth_ratio': depth_ratio
            }
        except Exception as e:
            self.logger.error(f"Error calculating market depth: {e}")
            return {}

    def get_market_metrics(self, symbol: str) -> Dict[str, float]:
        """Get various market metrics."""
        try:
            # Get 24h ticker
            ticker = self.client.get_ticker(symbol=symbol)
            
            # Get market depth
            depth = self.get_market_depth(symbol)
            
            # Get recent trades
            trades = self.get_recent_trades(symbol, limit=100)
            
            # Calculate metrics
            volume_24h = float(ticker['volume'])
            price_change_24h = float(ticker['priceChangePercent'])
            high_24h = float(ticker['highPrice'])
            low_24h = float(ticker['lowPrice'])
            
            # Calculate trade metrics
            trade_volume = trades['quoteQty'].sum()
            avg_trade_size = trade_volume / len(trades) if len(trades) > 0 else 0
            
            return {
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'spread': depth.get('spread', 0),
                'spread_percentage': depth.get('spread_percentage', 0),
                'depth_ratio': depth.get('depth_ratio', 0),
                'trade_volume': trade_volume,
                'avg_trade_size': avg_trade_size
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market metrics: {e}")
            return {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        try:
            self.logger.info(f"Starting add_technical_indicators. Input DataFrame shape: {df.shape}")
            self.logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
            self.logger.info(f"Input DataFrame dtypes:\n{df.dtypes}")

            if df.empty:
                self.logger.error("Empty DataFrame provided to add_technical_indicators")
                return df

            # Make a copy to avoid warnings
            df = df.copy()
            self.logger.info("Created DataFrame copy")

            # Verify required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns for indicators: {missing}")
                self.logger.error(f"Available columns: {df.columns.tolist()}")
                return df

            # Ensure all columns are numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                self.logger.info(f"Converted {col} to numeric. Sample value: {df[col].iloc[0] if not df.empty else 'N/A'}")

            # Drop any rows with NaN values
            original_shape = df.shape
            df = df.dropna(subset=required_columns)
            if df.shape != original_shape:
                self.logger.warning(f"Dropped {original_shape[0] - df.shape[0]} rows with NaN values")

            if df.empty:
                self.logger.error("No valid data after cleaning")
                return df

            self.logger.info("Starting to calculate technical indicators")

            # Calculate moving averages
            df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            df['EMA_20'] = df['close'].ewm(span=20, min_periods=1, adjust=False).mean()
            self.logger.info("Calculated moving averages")

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            self.logger.info("Calculated RSI")

            # Calculate MACD
            exp1 = df['close'].ewm(span=12, min_periods=1, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, min_periods=1, adjust=False).mean()
            df['MACD_12_26_9'] = exp1 - exp2
            df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, min_periods=1, adjust=False).mean()
            df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
            self.logger.info("Calculated MACD")

            # Calculate Bollinger Bands
            df['BBM_20_2.0'] = df['close'].rolling(window=20, min_periods=1).mean()
            std = df['close'].rolling(window=20, min_periods=1).std()
            df['BBU_20_2.0'] = df['BBM_20_2.0'] + (std * 2)
            df['BBL_20_2.0'] = df['BBM_20_2.0'] - (std * 2)
            self.logger.info("Calculated Bollinger Bands")

            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATRr_14'] = true_range.rolling(window=14, min_periods=1).mean()
            self.logger.info("Calculated ATR")

            # Calculate OBV (On Balance Volume)
            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            self.logger.info("Calculated OBV")

            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            self.logger.info("Filled NaN values")

            # Verify all required columns are present
            required_indicators = [
                'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14',
                'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                'ATRr_14', 'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', 'OBV'
            ]
            if not all(col in df.columns for col in required_indicators):
                missing = [col for col in required_indicators if col not in df.columns]
                self.logger.error(f"Missing technical indicators: {missing}")
                self.logger.error(f"Available columns: {df.columns.tolist()}")
                return df

            self.logger.info(f"Successfully added all technical indicators. Final shape: {df.shape}")
            self.logger.info(f"Final columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            self.logger.error(f"DataFrame state at error: shape={df.shape}, columns={df.columns.tolist()}")
            return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training with improved error handling."""
        try:
            if df.empty:
                self.logger.error("Empty dataframe provided")
                return pd.DataFrame(), pd.DataFrame()

            # Ensure all required columns exist
            required_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14',
                'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                'ATRr_14', 'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0',
                'OBV'
            ]

            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame(), pd.DataFrame()

            # Calculate future returns (target variable)
            df['future_return'] = df['close'].pct_change(periods=1).shift(-1)

            # Remove any rows with NaN values
            df = df.dropna()

            if df.empty:
                self.logger.error("No valid data after cleaning")
                return pd.DataFrame(), pd.DataFrame()

            # Prepare features (X) and target (y)
            X = df[required_columns].copy()
            y = df['future_return'].copy()

            # Normalize features
            for col in X.columns:
                if col != 'volume':  # Don't normalize volume
                    mean = X[col].mean()
                    std = X[col].std()
                    if std != 0:
                        X[col] = (X[col] - mean) / std

            self.logger.info(f"Successfully prepared training data. X shape: {X.shape}, y shape: {y.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    