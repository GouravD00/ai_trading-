from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

class ProviderBase(ABC):
    """
    Abstract base class for data providers with enhanced error handling
    and risk management features
    """
    
    def __init__(self, name: str, rate_limit: int = 5):
        self.name = name
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.connection_failures = 0
        self.max_retries = 3
        
    def _rate_limit_check(self):
        """Enforce rate limiting to prevent API abuse"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_diff < min_interval:
            sleep_time = min_interval - time_diff
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _handle_request_error(self, error: Exception, symbol: str) -> None:
        """Handle API request errors with exponential backoff"""
        self.connection_failures += 1
        self.logger.error(f"Request failed for {symbol}: {str(error)}")
        
        if self.connection_failures <= self.max_retries:
            backoff_time = 2 ** self.connection_failures
            self.logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
        else:
            self.logger.critical(f"Max retries exceeded for {self.name} provider")
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate incoming data for quality and completeness"""
        if data is None or data.empty:
            self.logger.warning(f"Empty data received for {symbol}")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return False
        
        # Check for data anomalies
        if (data['high'] < data['low']).any():
            self.logger.error(f"Data anomaly detected for {symbol}: high < low")
            return False
        
        if (data['close'] > data['high']).any() or (data['close'] < data['low']).any():
            self.logger.error(f"Data anomaly detected for {symbol}: close outside high/low range")
            return False
        
        # Check for excessive gaps or spikes
        price_changes = data['close'].pct_change().dropna()
        if (abs(price_changes) > 0.2).any():  # 20% price change threshold
            self.logger.warning(f"Large price movements detected for {symbol}")
        
        return True
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add essential technical indicators for risk assessment"""
        try:
            # Simple Moving Averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            # Average True Range (ATR) for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate risk metrics for the symbol"""
        try:
            if len(data) < 30:
                return {'risk_level': 'high', 'reason': 'insufficient_data'}
            
            returns = data['close'].pct_change().dropna()
            
            # Volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            excess_returns = returns.mean() - 0.02/252
            sharpe_ratio = excess_returns / returns.std() if returns.std() > 0 else 0
            
            # Risk level classification
            risk_level = 'low'
            if volatility > 0.3 or max_drawdown < -0.2:
                risk_level = 'high'
            elif volatility > 0.2 or max_drawdown < -0.1:
                risk_level = 'medium'
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'risk_level': risk_level,
                'avg_volume': data['volume'].mean(),
                'price_stability': 1 - volatility  # Inverse of volatility
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'risk_level': 'high', 'reason': 'calculation_error'}
    
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        pass
    
    @abstractmethod
    def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time data for a symbol"""
        pass
    
    @abstractmethod
    def get_market_status(self) -> Dict:
        """Get current market status"""
        pass
    
    def health_check(self) -> Dict:
        """Check provider health status"""
        return {
            'provider': self.name,
            'status': 'healthy' if self.connection_failures < self.max_retries else 'degraded',
            'connection_failures': self.connection_failures,
            'last_request': self.last_request_time,
            'timestamp': datetime.now().isoformat()
        }