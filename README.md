# AI-Powered Trading Bot

This trading bot uses artificial intelligence and deep learning to predict cryptocurrency price movements and execute trades on Binance. It implements the Silver Bullet strategy and aims for a daily profit of 1-2% while maintaining strict risk management.

## Features

- Deep Learning-based price prediction using LSTM networks
- Technical analysis indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Risk management with stop-loss and take-profit levels
- Trading time restrictions (Silver Bullet strategy)
- Multiple trading pairs support
- Real-time market data processing
- Comprehensive logging system

## Requirements

- Python 3.8+
- Binance account with API access
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Configuration

The bot's behavior can be customized by modifying the parameters in `config.py`:

- Trading pairs
- Capital allocation
- Profit targets
- Risk management parameters
- Trading hours
- Model parameters

## Usage

1. Ensure your `.env` file is properly configured with your Binance API credentials.

2. Run the bot:
```bash
python trading_bot.py
```

The bot will:
- Initialize and train models for each trading pair
- Start monitoring the market during trading hours
- Execute trades based on predictions and risk management rules
- Log all activities to `trading_bot.log`

## Risk Warning

Trading cryptocurrencies involves significant risk. This bot is provided for educational purposes only. Always:
- Start with small amounts
- Monitor the bot's performance
- Never invest more than you can afford to lose
- Test thoroughly in a paper trading environment first

## Performance

The bot aims for:
- 95% prediction accuracy
- 1-2% daily profit target
- Strict risk management with 1% stop-loss and 2% take-profit levels

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 