import requests
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
import os
from dotenv import load_dotenv
import tweepy
import numpy as np
from collections import defaultdict

class SentimentAnalysis:
    def __init__(self):
        load_dotenv()
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        # Initialize Twitter API
        auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
        auth.set_access_token(self.twitter_access_token, self.twitter_access_token_secret)
        self.twitter_api = tweepy.API(auth)
        
    def analyze_twitter_sentiment(self, symbol: str, count: int = 100) -> Dict[str, float]:
        """Analyze sentiment from Twitter."""
        try:
            # Search tweets
            tweets = self.twitter_api.search_tweets(
                q=f"#{symbol} OR ${symbol}",
                lang="en",
                count=count,
                tweet_mode="extended"
            )
            
            # Calculate sentiment
            sentiments = []
            for tweet in tweets:
                text = tweet.full_text
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            if not sentiments:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'NEUTRAL',
                    'tweet_count': 0
                }
            
            avg_sentiment = np.mean(sentiments)
            
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
                'tweet_count': len(sentiments)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing Twitter sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'tweet_count': 0
            }
    
    def analyze_reddit_sentiment(self, symbol: str, subreddits: List[str] = ['CryptoCurrency', 'CryptoMarkets']) -> Dict[str, float]:
        """Analyze sentiment from Reddit."""
        try:
            # Set up Reddit API
            auth = requests.auth.HTTPBasicAuth(self.reddit_client_id, self.reddit_client_secret)
            headers = {'User-Agent': 'TradingBot/1.0'}
            
            sentiments = []
            total_posts = 0
            
            for subreddit in subreddits:
                # Get posts
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': symbol,
                    'limit': 100,
                    'sort': 'relevance',
                    't': 'day'
                }
                
                response = requests.get(url, headers=headers, params=params)
                posts = response.json()['data']['children']
                
                # Analyze sentiment
                for post in posts:
                    title = post['data']['title']
                    text = post['data']['selftext']
                    
                    # Combine title and text
                    content = f"{title} {text}"
                    blob = TextBlob(content)
                    sentiments.append(blob.sentiment.polarity)
                    total_posts += 1
            
            if not sentiments:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'NEUTRAL',
                    'post_count': 0
                }
            
            avg_sentiment = np.mean(sentiments)
            
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
                'post_count': total_posts
            }
            
        except Exception as e:
            logging.error(f"Error analyzing Reddit sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'post_count': 0
            }
    
    def analyze_google_trends(self, symbol: str) -> Dict[str, float]:
        """Analyze Google Trends data."""
        try:
            # Get Google Trends data
            url = f"https://trends.google.com/trends/api/dailytrends"
            params = {
                'hl': 'en-US',
                'tz': '-120',
                'geo': 'US',
                'ns': '15'
            }
            
            response = requests.get(url, params=params)
            trends_data = response.json()
            
            # Find symbol in trends
            symbol_trend = None
            for trend in trends_data['default']['trendingSearchesDays'][0]['trendingSearches']:
                if symbol.lower() in trend['title']['query'].lower():
                    symbol_trend = trend
                    break
            
            if not symbol_trend:
                return {
                    'trend_score': 0,
                    'trend_label': 'NEUTRAL'
                }
            
            # Calculate trend score
            trend_score = symbol_trend.get('formattedTraffic', '0').replace('+', '').replace('K', '000')
            trend_score = float(trend_score)
            
            # Determine trend label
            if trend_score > 100000:
                label = 'HIGH'
            elif trend_score > 10000:
                label = 'MEDIUM'
            else:
                label = 'LOW'
            
            return {
                'trend_score': trend_score,
                'trend_label': label
            }
            
        except Exception as e:
            logging.error(f"Error analyzing Google Trends: {e}")
            return {
                'trend_score': 0,
                'trend_label': 'NEUTRAL'
            }
    
    def get_overall_sentiment(self, symbol: str) -> Dict[str, any]:
        """Get overall market sentiment from all sources."""
        twitter_sentiment = self.analyze_twitter_sentiment(symbol)
        reddit_sentiment = self.analyze_reddit_sentiment(symbol)
        google_trends = self.analyze_google_trends(symbol)
        
        # Calculate weighted sentiment score
        weights = {
            'twitter': 0.4,
            'reddit': 0.4,
            'google_trends': 0.2
        }
        
        weighted_score = (
            twitter_sentiment['sentiment_score'] * weights['twitter'] +
            reddit_sentiment['sentiment_score'] * weights['reddit'] +
            (google_trends['trend_score'] / 100000) * weights['google_trends']
        )
        
        # Determine overall sentiment
        if weighted_score > 0.2:
            overall_sentiment = 'BULLISH'
        elif weighted_score < -0.2:
            overall_sentiment = 'BEARISH'
        else:
            overall_sentiment = 'NEUTRAL'
        
        return {
            'twitter_sentiment': twitter_sentiment,
            'reddit_sentiment': reddit_sentiment,
            'google_trends': google_trends,
            'weighted_score': weighted_score,
            'overall_sentiment': overall_sentiment
        }
    
    def get_sentiment_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get historical sentiment data."""
        try:
            sentiments = []
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                sentiment = self.get_overall_sentiment(symbol)
                sentiment['date'] = date
                sentiments.append(sentiment)
            
            df = pd.DataFrame(sentiments)
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error getting sentiment history: {e}")
            return pd.DataFrame() 