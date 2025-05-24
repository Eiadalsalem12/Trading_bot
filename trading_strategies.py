from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from tradingview_ta import TA_Handler, Interval
from newsapi import NewsApiClient
import yfinance as yf

class TradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config.get('NEWS_API_KEY'))
        self.strategies = {
            'rsi': self.rsi_strategy,
            'macd': self.macd_strategy,
            'bollinger': self.bollinger_strategy,
            'silver_bullet': self.silver_bullet_strategy,
            'wave_analysis': self.wave_analysis_strategy,
            'time_analysis': self.time_analysis_strategy
        }

    def get_tradingview_analysis(self, symbol: str) -> Dict:
        """الحصول على تحليل TradingView"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener="crypto",
                exchange="BINANCE",
                interval=Interval.INTERVAL_1_HOUR
            )
            analysis = handler.get_analysis()
            return analysis
        except Exception as e:
            print(f"Error getting TradingView analysis: {e}")
            return None

    def rsi_strategy(self, symbol: str) -> Dict:
        """استراتيجية RSI باستخدام TradingView"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        rsi = analysis.indicators['RSI']
        signals = {
            'signal': 0,
            'confidence': 0
        }

        # شروط الشراء: RSI أقل من 30 (سوق مفرط في البيع)
        if rsi < 30:
            signals['signal'] = 1  # إشارة شراء
            signals['confidence'] = 0.8
        # شروط البيع: RSI أعلى من 70 (سوق مفرط في الشراء)
        elif rsi > 70:
            signals['signal'] = -1  # إشارة بيع
            signals['confidence'] = 0.8

        return signals

    def macd_strategy(self, symbol: str) -> Dict:
        """استراتيجية MACD باستخدام TradingView"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        macd = analysis.indicators['MACD.macd']
        macd_signal = analysis.indicators['MACD.signal']
        signals = {
            'signal': 0,
            'confidence': 0
        }

        # شروط الشراء: MACD يتجاوز خط الإشارة من الأسفل
        if macd > macd_signal:
            signals['signal'] = 1  # إشارة شراء
            signals['confidence'] = 0.7
        # شروط البيع: MACD يتجاوز خط الإشارة من الأعلى
        elif macd < macd_signal:
            signals['signal'] = -1  # إشارة بيع
            signals['confidence'] = 0.7

        return signals

    def bollinger_strategy(self, symbol: str) -> Dict:
        """استراتيجية Bollinger Bands باستخدام TradingView"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        close = analysis.indicators['close']
        bb_upper = analysis.indicators['BB.upperband']
        bb_lower = analysis.indicators['BB.lowerband']
        signals = {
            'signal': 0,
            'confidence': 0
        }

        # شروط الشراء: السعر يلامس الحد السفلي
        if close < bb_lower:
            signals['signal'] = 1  # إشارة شراء
            signals['confidence'] = 0.75
        # شروط البيع: السعر يلامس الحد العلوي
        elif close > bb_upper:
            signals['signal'] = -1  # إشارة بيع
            signals['confidence'] = 0.75

        return signals

    def silver_bullet_strategy(self, symbol: str) -> Dict:
        """استراتيجية Silver Bullet"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        signals = {
            'signal': 0,
            'confidence': 0
        }

        # شروط Silver Bullet
        rsi = analysis.indicators['RSI']
        macd = analysis.indicators['MACD.macd']
        macd_signal = analysis.indicators['MACD.signal']
        stoch_k = analysis.indicators['Stoch.K']
        stoch_d = analysis.indicators['Stoch.D']
        ema_20 = analysis.indicators['EMA20']
        ema_50 = analysis.indicators['EMA50']
        close = analysis.indicators['close']

        # شروط الشراء في Silver Bullet
        if (rsi < 30 and  # RSI مفرط في البيع
            macd > macd_signal and  # MACD إيجابي
            stoch_k < 20 and stoch_d < 20 and  # Stochastic مفرط في البيع
            close > ema_20 and ema_20 > ema_50):  # الاتجاه صعودي
            signals['signal'] = 1
            signals['confidence'] = 0.85

        # شروط البيع في Silver Bullet
        elif (rsi > 70 and  # RSI مفرط في الشراء
              macd < macd_signal and  # MACD سلبي
              stoch_k > 80 and stoch_d > 80 and  # Stochastic مفرط في الشراء
              close < ema_20 and ema_20 < ema_50):  # الاتجاه هبوطي
            signals['signal'] = -1
            signals['confidence'] = 0.85

        return signals

    def wave_analysis_strategy(self, symbol: str) -> Dict:
        """استراتيجية التحليل الموجي"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        signals = {
            'signal': 0,
            'confidence': 0
        }

        # مؤشرات التحليل الموجي
        rsi = analysis.indicators['RSI']
        macd = analysis.indicators['MACD.macd']
        macd_signal = analysis.indicators['MACD.signal']
        ema_20 = analysis.indicators['EMA20']
        ema_50 = analysis.indicators['EMA50']
        close = analysis.indicators['close']
        volume = analysis.indicators['volume']

        # شروط الشراء في التحليل الموجي
        if (rsi < 40 and  # RSI في منطقة التشبع البيعي
            macd > macd_signal and  # MACD إيجابي
            close > ema_20 and  # السعر فوق EMA20
            volume > analysis.indicators['volume'].mean()):  # حجم تداول مرتفع
            signals['signal'] = 1
            signals['confidence'] = 0.8

        # شروط البيع في التحليل الموجي
        elif (rsi > 60 and  # RSI في منطقة التشبع الشرائي
              macd < macd_signal and  # MACD سلبي
              close < ema_20 and  # السعر تحت EMA20
              volume > analysis.indicators['volume'].mean()):  # حجم تداول مرتفع
            signals['signal'] = -1
            signals['confidence'] = 0.8

        return signals

    def time_analysis_strategy(self, symbol: str) -> Dict:
        """استراتيجية التحليل الزمني"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'signal': 0, 'confidence': 0}

        signals = {
            'signal': 0,
            'confidence': 0
        }

        # مؤشرات التحليل الزمني
        rsi = analysis.indicators['RSI']
        macd = analysis.indicators['MACD.macd']
        macd_signal = analysis.indicators['MACD.signal']
        ema_20 = analysis.indicators['EMA20']
        ema_50 = analysis.indicators['EMA50']
        close = analysis.indicators['close']
        volume = analysis.indicators['volume']

        # شروط الشراء في التحليل الزمني
        if (rsi < 35 and  # RSI في منطقة التشبع البيعي
            macd > macd_signal and  # MACD إيجابي
            close > ema_20 and  # السعر فوق EMA20
            volume > analysis.indicators['volume'].mean() * 1.5):  # حجم تداول مرتفع جداً
            signals['signal'] = 1
            signals['confidence'] = 0.75

        # شروط البيع في التحليل الزمني
        elif (rsi > 65 and  # RSI في منطقة التشبع الشرائي
              macd < macd_signal and  # MACD سلبي
              close < ema_20 and  # السعر تحت EMA20
              volume > analysis.indicators['volume'].mean() * 1.5):  # حجم تداول مرتفع جداً
            signals['signal'] = -1
            signals['confidence'] = 0.75

        return signals
    
    def get_all_signals(self, symbol: str) -> Dict:
        """الحصول على إشارات من جميع الاستراتيجيات"""
        signals = {}
        for strategy_name in self.strategies:
            signals[strategy_name] = self.strategies[strategy_name](symbol)
        return signals

class SwingTradingStrategy(TradingStrategy):
    def analyze(self, symbol: str) -> Dict:
        """تحليل استراتيجية التداول المتأرجح"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'buy': False, 'sell': False, 'confidence': 0}
        
        signals = {
            'buy': False,
            'sell': False,
            'confidence': 0.0
        }
        
        # شروط الشراء
        if (analysis.indicators['RSI'] < 30 and 
            analysis.indicators['MACD.macd'] > analysis.indicators['MACD.signal']):
            signals['buy'] = True
            signals['confidence'] = 0.8
        
        # شروط البيع
        elif (analysis.indicators['RSI'] > 70 and 
              analysis.indicators['MACD.macd'] < analysis.indicators['MACD.signal']):
            signals['sell'] = True
            signals['confidence'] = 0.8
            
        return signals

class MeanReversionStrategy(TradingStrategy):
    def analyze(self, symbol: str) -> Dict:
        """تحليل استراتيجية التداول المتوسط"""
        analysis = self.get_tradingview_analysis(symbol)
        if not analysis:
            return {'buy': False, 'sell': False, 'confidence': 0}
        
        signals = {
            'buy': False,
            'sell': False,
            'confidence': 0.0
        }
        
        # شروط الشراء
        if (analysis.indicators['close'] < analysis.indicators['BB.lowerband'] and
            analysis.indicators['RSI'] < 30):
            signals['buy'] = True
            signals['confidence'] = 0.7
        
        # شروط البيع
        elif (analysis.indicators['close'] > analysis.indicators['BB.upperband'] and
              analysis.indicators['RSI'] > 70):
            signals['sell'] = True
            signals['confidence'] = 0.7
            
        return signals

class NewsBasedStrategy(TradingStrategy):
    def analyze(self, symbol: str) -> Dict:
        """تحليل استراتيجية التداول على أساس الأخبار"""
        signals = {
            'buy': False,
            'sell': False,
            'confidence': 0.0
        }
        
        # جلب الأخبار
        news = self.newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='relevancy'
        )
        
        # تحليل الأخبار
        sentiment_score = 0
        for article in news['articles'][:5]:
            # هنا يمكن إضافة تحليل المشاعر باستخدام NLP
            pass
        
        if sentiment_score > 0.5:
            signals['buy'] = True
            signals['confidence'] = 0.6
        elif sentiment_score < -0.5:
            signals['sell'] = True
            signals['confidence'] = 0.6
            
        return signals

class FundamentalAnalysisStrategy(TradingStrategy):
    def analyze(self, symbol: str) -> Dict:
        """تحليل استراتيجية التداول على أساس التحليل الأساسي"""
        signals = {
            'buy': False,
            'sell': False,
            'confidence': 0.0
        }
        
        try:
            # جلب البيانات الأساسية
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # تحليل البيانات الأساسية
            if (info.get('forwardPE', 0) < 15 and
                info.get('dividendYield', 0) > 0.03 and
                info.get('debtToEquity', 0) < 1):
                signals['buy'] = True
                signals['confidence'] = 0.7
            elif (info.get('forwardPE', 0) > 30 or
                  info.get('debtToEquity', 0) > 2):
                signals['sell'] = True
                signals['confidence'] = 0.7
                
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
            
        return signals 