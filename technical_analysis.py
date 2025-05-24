import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple, List
import logging

class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            if df.empty:
                logging.warning("Empty dataframe provided for technical analysis")
                return df

            # Ensure we have enough data points
            if len(df) < 50:
                logging.error(f"Insufficient data points: {len(df)}. Need at least 50 points.")
                return df

            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop any rows with NaN values after conversion
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

            # Check if we still have enough data points after cleaning
            if len(df) < 50:
                logging.error(f"Not enough valid data points after cleaning: {len(df)}. Need at least 50 points.")
                return df

            # Trend Indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # Volatility Indicators
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14
            )
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Volume Indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            # Drop any rows with NaN values in indicators
            df = df.dropna()

            # Final check for sufficient data points
            if len(df) < 50:
                logging.error(f"Not enough data points after calculating indicators: {len(df)}. Need at least 50 points.")
                return df

            return df

        except Exception as e:
            logging.error(f"Error adding technical indicators: {str(e)}")
            return df

    @staticmethod
    def detect_trend(df: pd.DataFrame) -> str:
        """Detect the current market trend."""
        try:
            if df.empty:
                return "UNKNOWN"
                
            # Use SMA crossover
            if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
                return "UPTREND"
            elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            logging.error(f"Error detecting trend: {e}")
            return "UNKNOWN"

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels."""
        try:
            if df.empty:
                return 0.0, 0.0
                
            # Use recent highs and lows
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            return recent_low, recent_high
            
        except Exception as e:
            logging.error(f"Error calculating support/resistance: {e}")
            return 0.0, 0.0

    @staticmethod
    def analyze_volume(df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trading volume."""
        try:
            if df.empty:
                return {}
                
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            return {
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logging.error(f"Error analyzing volume: {e}")
            return {}

    @staticmethod
    def detect_divergence(df: pd.DataFrame) -> Dict[str, bool]:
        """Detect price and indicator divergences."""
        try:
            if df.empty:
                return {}
                
            # RSI divergence
            price_higher = df['close'].iloc[-1] > df['close'].iloc[-2]
            rsi_higher = df['rsi'].iloc[-1] > df['rsi'].iloc[-2]
            
            # MACD divergence
            macd_higher = df['macd'].iloc[-1] > df['macd'].iloc[-2]
            
            return {
                'rsi_bullish_divergence': not price_higher and rsi_higher,
                'rsi_bearish_divergence': price_higher and not rsi_higher,
                'macd_bullish_divergence': not price_higher and macd_higher,
                'macd_bearish_divergence': price_higher and not macd_higher
            }
            
        except Exception as e:
            logging.error(f"Error detecting divergence: {e}")
            return {}

    @staticmethod
    def calculate_volatility(df: pd.DataFrame) -> float:
        """Calculate market volatility."""
        try:
            if df.empty:
                return 0.0
                
            returns = df['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)  # Annualized volatility
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {e}")
            return 0.0

    @staticmethod
    def get_market_conditions(df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive market conditions."""
        try:
            if df.empty:
                return {}
                
            trend = TechnicalAnalysis.detect_trend(df)
            support, resistance = TechnicalAnalysis.calculate_support_resistance(df)
            volume_analysis = TechnicalAnalysis.analyze_volume(df)
            divergence = TechnicalAnalysis.detect_divergence(df)
            volatility = TechnicalAnalysis.calculate_volatility(df)
            
            return {
                'trend': trend,
                'support': support,
                'resistance': resistance,
                'volume_analysis': volume_analysis,
                'divergence': divergence,
                'volatility': volatility,
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
                'macd': df['macd'].iloc[-1] if 'macd' in df.columns else None
            }
            
        except Exception as e:
            logging.error(f"Error getting market conditions: {e}")
            return {}

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on technical indicators."""
        try:
            if df.empty:
                return df
                
            # Initialize signal column
            df['signal'] = 0
            
            # Trend following signals
            df.loc[df['sma_20'] > df['sma_50'], 'signal'] += 1
            df.loc[df['sma_20'] < df['sma_50'], 'signal'] -= 1
            
            # RSI signals
            df.loc[df['rsi'] < 30, 'signal'] += 1  # Oversold
            df.loc[df['rsi'] > 70, 'signal'] -= 1  # Overbought
            
            # MACD signals
            df.loc[df['macd'] > df['macd_signal'], 'signal'] += 1
            df.loc[df['macd'] < df['macd_signal'], 'signal'] -= 1
            
            # Bollinger Bands signals
            df.loc[df['close'] < df['bb_lower'], 'signal'] += 1  # Oversold
            df.loc[df['close'] > df['bb_upper'], 'signal'] -= 1  # Overbought
            
            # Normalize signals to -1, 0, 1
            df['signal'] = df['signal'].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating signals: {e}")
            return df

if __name__ == "__main__":
    print("تم تشغيل ملف technical_analysis.py بنجاح") 