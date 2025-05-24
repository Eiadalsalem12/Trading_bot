import requests
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
import os
from dotenv import load_dotenv

class FundamentalAnalysis:
    def __init__(self):
        load_dotenv()
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.crypto_compare_key = os.getenv('CRYPTO_COMPARE_KEY')
        
    def get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get news sentiment for a given symbol."""
        try:
            # Get news from CryptoCompare
            url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={self.crypto_compare_key}"
            response = requests.get(url)
            news_data = response.json()
            
            # Filter news for the specific symbol
            symbol_news = [
                news for news in news_data['Data']
                if symbol.lower() in news['title'].lower() or symbol.lower() in news['body'].lower()
            ]
            
            # Calculate sentiment
            sentiments = []
            for news in symbol_news:
                blob = TextBlob(news['title'] + " " + news['body'])
                sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'NEUTRAL',
                    'news_count': 0
                }
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Determine sentiment label
            if avg_sentiment > 0.2:
                label = 'BULLISH'
            elif avg_sentiment < -0.2:
                label = 'BEARISH'
            else:
                label = 'NEUTRAL'
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_label': label,
                'news_count': len(sentiments)
            }
            
        except Exception as e:
            logging.error(f"Error getting news sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'news_count': 0
            }
    
    def get_market_events(self) -> List[Dict]:
        """Get upcoming market events."""
        try:
            # Get events from CoinMarketCal
            url = "https://api.coinmarketcal.com/v1/events"
            headers = {
                'x-api-key': self.crypto_compare_key
            }
            response = requests.get(url, headers=headers)
            events_data = response.json()
            
            # Filter and format events
            upcoming_events = []
            for event in events_data:
                event_date = datetime.fromisoformat(event['date'])
                if event_date > datetime.now():
                    upcoming_events.append({
                        'title': event['title'],
                        'date': event_date,
                        'importance': event['importance'],
                        'symbols': event['symbols']
                    })
            
            return sorted(upcoming_events, key=lambda x: x['date'])
            
        except Exception as e:
            logging.error(f"Error getting market events: {e}")
            return []
    
    def analyze_market_impact(self, symbol: str) -> Dict[str, any]:
        """Analyze fundamental market impact."""
        sentiment = self.get_news_sentiment(symbol)
        events = self.get_market_events()
        
        # Filter events for the symbol
        symbol_events = [
            event for event in events
            if symbol in event['symbols']
        ]
        
        # Calculate event impact
        event_impact = 0
        for event in symbol_events:
            if event['importance'] == 'high':
                event_impact += 1
            elif event['importance'] == 'medium':
                event_impact += 0.5
        
        return {
            'news_sentiment': sentiment,
            'upcoming_events': symbol_events,
            'event_impact': event_impact,
            'overall_impact': (
                'HIGH' if (sentiment['sentiment_score'] > 0.3 and event_impact > 1) or
                          (sentiment['sentiment_score'] < -0.3 and event_impact > 1)
                else 'LOW'
            )
        }
    
    def get_market_overview(self) -> Dict[str, any]:
        """Get overall market overview."""
        try:
            # Get market data from CoinGecko
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url)
            market_data = response.json()
            
            return {
                'total_market_cap': market_data['data']['total_market_cap']['usd'],
                'total_volume': market_data['data']['total_volume']['usd'],
                'market_cap_change_24h': market_data['data']['market_cap_change_percentage_24h_usd'],
                'btc_dominance': market_data['data']['market_cap_percentage']['btc'],
                'eth_dominance': market_data['data']['market_cap_percentage']['eth']
            }
            
        except Exception as e:
            logging.error(f"Error getting market overview: {e}")
            return {} 