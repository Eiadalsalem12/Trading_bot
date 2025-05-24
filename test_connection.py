from binance.client import Client
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_binance_connection():
    """Test the connection to Binance API."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API keys
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logging.error("API keys not found in .env file")
            return False
            
        # Initialize client
        client = Client(api_key, api_secret)
        
        # Test connection by getting account info
        account = client.get_account()
        logging.info("Successfully connected to Binance API")
        logging.info(f"Account status: {account['accountType']}")
        
        # Get USDT balance
        usdt_balance = float([asset for asset in account['balances'] if asset['asset'] == 'USDT'][0]['free'])
        logging.info(f"USDT Balance: {usdt_balance}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error connecting to Binance API: {e}")
        return False

if __name__ == "__main__":
    test_binance_connection() 