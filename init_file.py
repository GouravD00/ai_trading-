# datahub/__init__.py
"""
AI Trader DataHub Package
Comprehensive trading system with enhanced risk management
"""

__version__ = "1.0.0"
__author__ = "AI Trading System"

# Import main components
from .provider_base import ProviderBase
from .yfinance_provider import YFinanceProvider
from .alpha_vantage_provider import AlphaVantageProvider
from .data_manager import DataManager

# Export main classes
__all__ = [
    'ProviderBase',
    'YFinanceProvider', 
    'AlphaVantageProvider',
    'DataManager'
]

# Configuration constants
DEFAULT_CONFIG = {
    'risk_management': {
        'max_position_size': 0.02,  # 2% max position size
        'stop_loss': 0.05,          # 5% stop loss
        'take_profit': 0.10,        # 10% take profit
        'max_daily_loss': 0.03,     # 3% max daily loss
        'max_open_positions': 5,    # Max 5 open positions
    },
    'data_sources': {
        'primary': 'yfinance',
        'backup': 'alpha_vantage',
        'update_frequency': 300,    # 5 minutes
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}