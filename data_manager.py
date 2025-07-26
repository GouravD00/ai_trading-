import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import threading
import time
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from .provider_base import ProviderBase
from .yfinance_provider import YFinanceProvider
from .alpha_vantage_provider import AlphaVantageProvider

class DataManager:
    """
    Centralized data management system with enhanced risk controls,
    failover capabilities, and intelligent data aggregation
    """
    
    def __init__(self, alpha_vantage_key: str = "demo", cache_db_path: str = "trading_cache.db"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize data providers
        self.providers = {
            'yfinance': YFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider(alpha_vantage_key)
        }
        
        self.primary_provider = 'yfinance'
        self.backup_provider = 'alpha_vantage'
        
        # Cache and storage
        self.cache_db_path = cache_db_path
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes default
        
        # Risk management
        self.risk_thresholds = {
            'max_volatility': 0.4,      # 40% annualized volatility
            'max_drawdown': -0.25,      # 25% maximum drawdown
            'min_liquidity': 100000,    # Minimum daily volume
            'max_beta': 2.0,            # Maximum beta
            'min_market_cap': 1e8       # $100M minimum market cap
        }
        
        # Performance tracking
        self.provider_performance = {
            'yfinance': {'success_rate': 1.0, 'avg_response_time': 0.0, 'last_success': datetime.now()},
            'alpha_vantage': {'success_rate': 1.0, 'avg_response_time': 0.0, 'last_success': datetime.now()}
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.data_lock = threading.Lock()
        
        # Initialize database
        self._init_cache_db()
        
        # Portfolio tracking
        self.portfolio = {}
        self.watchlist = set()
        self.risk_alerts = []
        
        self.logger.info("DataManager initialized with enhanced risk management")
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Create tables for caching
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    provider TEXT,
                    timestamp TEXT,
                    data TEXT,
                    expiry TEXT,
                    PRIMARY KEY (symbol, provider)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    symbol TEXT PRIMARY KEY,
                    risk_level TEXT,
                    volatility REAL,
                    beta REAL,
                    max_drawdown REAL,
                    last_updated TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    signal_type TEXT,
                    strength REAL,
                    timestamp TEXT,
                    conditions TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing cache database: {str(e)}")
    
    def get_stock_data(self, symbol: str, period: str = "1y", 
                      force_refresh: bool = False, 
                      include_risk_analysis: bool = True) -> Optional[pd.DataFrame]:
        """
        Get comprehensive stock data with risk analysis and provider failover
        """
        cache_key = f"{symbol}_{period}"
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_key):
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Returning cached data for {symbol}")
                return cached_data
        
        # Try primary provider first
        primary_data = self._get_data_from_provider(self.primary_provider, symbol, period)
        
        if primary_data is not None:
            # Enhance with backup provider data if available
            backup_data = self._get_data_from_provider(self.backup_provider, symbol, period)
            
            if backup_data is not None:
                primary_data = self._merge_provider_data(primary_data, backup_data)
            
            # Add comprehensive risk analysis
            if include_risk_analysis:
                primary_data = self._add_comprehensive_risk_analysis(primary_data, symbol)
            
            # Cache the result
            self._cache_data(cache_key, primary_data)
            
            # Update risk database
            self._update_risk_database(symbol, primary_data)
            
            return primary_data
        
        # Fallback to backup provider
        self.logger.warning(f"Primary provider failed for {symbol}, trying backup")
        backup_data = self._get_data_from_provider(self.backup_provider, symbol, period)
        
        if backup_data is not None:
            if include_risk_analysis:
                backup_data = self._add_comprehensive_risk_analysis(backup_data, symbol)
            
            self._cache_data(cache_key, backup_data)
            self._update_risk_database(symbol, backup_data)
            return backup_data
        
        self.logger.error(f"All providers failed for {symbol}")
        return None
    
    def _get_data_from_provider(self, provider_name: str, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get data from specific provider with performance tracking"""
        start_time = time.time()
        
        try:
            provider = self.providers[provider_name]
            data = provider.get_historical_data(symbol, period)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_provider_performance(provider_name, True, response_time)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Provider {provider_name} failed for {symbol}: {str(e)}")
            self._update_provider_performance(provider_name, False, 0)
            return None
    
    def _update_provider_performance(self, provider_name: str, success: bool, response_time: float):
        """Update provider performance metrics"""
        with self.data_lock:
            perf = self.provider_performance[provider_name]
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * (1 if success else 0)
            
            # Update average response time
            if success:
                perf['avg_response_time'] = (1 - alpha) * perf['avg_response_time'] + alpha * response_time
                perf['last_success'] = datetime.now()
    
    def _merge_provider_data(self, primary_data: pd.DataFrame, backup_data: pd.DataFrame) -> pd.DataFrame:
        """Intelligently merge data from multiple providers"""
        try:
            # Merge on index (datetime)
            merged = primary_data.copy()
            
            # Fill missing values with backup data
            for column in backup_data.columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(backup_data[column])
                else:
                    merged[f"{column}_backup"] = backup_data[column]
            
            # Combine risk metrics
            if hasattr(primary_data, 'attrs') and hasattr(backup_data, 'attrs'):
                primary_risk = primary_data.attrs.get('risk_metrics', {})
                backup_risk = backup_data.attrs.get('risk_metrics', {})
                
                # Use more conservative risk assessment
                combined_risk = primary_risk.copy()
                if backup_risk.get('risk_level') == 'high' or primary_risk.get('risk_level') == 'high':
                    combined_risk['risk_level'] = 'high'
                
                merged.attrs = primary_data.attrs.copy()
                merged.attrs['risk_metrics'] = combined_risk
                merged.attrs['data_sources'] = [primary_data.attrs.get('provider'), backup_data.attrs.get('provider')]
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging provider data: {str(e)}")
            return primary_data
    
    def _add_comprehensive_risk_analysis(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive risk analysis to stock data"""
        try:
            # Market regime analysis
            data = self._add_market_regime_analysis(data)
            
            # Volatility clustering
            data = self._add_volatility_clustering(data)
            
            # Support and resistance levels
            data = self._add_support_resistance_levels(data)
            
            # Risk-adjusted momentum
            data = self._add_risk_adjusted_momentum(data)
            
            # Liquidity analysis
            data = self._add_liquidity_analysis(data)
            
            # Correlation with market
            data = self._add_market_correlation(data)
            
            # Options flow indicators (if available)
            data = self._add_options_indicators(data, symbol)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive risk analysis for {symbol}: {str(e)}")
            return data
    
    def _add_market_regime_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify market regimes (bull, bear, sideways)"""
        try:
            # Use 50-day and 200-day moving averages
            sma_50 = data['close'].rolling(50).mean()
            sma_200 = data['close'].rolling(200).mean()
            
            # Define regimes
            conditions = [
                (data['close'] > sma_50) & (sma_50 > sma_200),  # Bull market
                (data['close'] < sma_50) & (sma_50 < sma_200),  # Bear market
            ]
            choices = ['bull', 'bear']
            
            data['market_regime'] = np.select(conditions, choices, default='sideways')
            
            # Regime strength
            regime_strength = abs(data['close'] - sma_50) / data['close']
            data['regime_strength'] = regime_strength
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {str(e)}")
            return data
    
    def _add_volatility_clustering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering analysis"""
        try:
            returns = data['close'].pct_change()
            
            # Sharpe ratio momentum
            rolling_returns = returns.rolling(20).mean()
            rolling_volatility = returns.rolling(20).std()
            data['sharpe_momentum'] = rolling_returns / rolling_volatility
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_vol = pd.Series(index=returns.index, dtype=float)
            
            for i in range(20, len(returns)):
                window_returns = returns.iloc[i-20:i]
                downside_window = window_returns[window_returns < 0]
                if len(downside_window) > 0:
                    downside_vol.iloc[i] = downside_window.std()
                else:
                    downside_vol.iloc[i] = 0.001  # Small value to avoid division by zero
            
            data['sortino_ratio'] = rolling_returns / downside_vol
            
            # Calmar ratio (return/max drawdown)
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(252).max()  # 1 year rolling max
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(252).min()
            
            data['calmar_ratio'] = rolling_returns / abs(max_drawdown)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in risk-adjusted momentum: {str(e)}")
            return data
    
    def _add_liquidity_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity analysis"""
        try:
            # Volume-based liquidity measures
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # Amihud illiquidity measure
            returns = abs(data['close'].pct_change())
            dollar_volume = data['close'] * data['volume']
            data['amihud_illiquidity'] = returns / (dollar_volume / 1e6)  # Scale by millions
            
            # Liquidity risk flags
            data['low_liquidity'] = (data['volume_ratio'] < 0.5) | (data['amihud_illiquidity'] > data['amihud_illiquidity'].quantile(0.95))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in liquidity analysis: {str(e)}")
            return data
    
    def _add_market_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market correlation analysis"""
        try:
            # Get market data (SPY as proxy)
            market_data = self._get_market_benchmark()
            
            if market_data is not None:
                # Align data
                aligned_data = data.join(market_data[['close']], rsuffix='_market', how='inner')
                
                if len(aligned_data) > 50:
                    # Rolling correlation
                    stock_returns = aligned_data['close'].pct_change()
                    market_returns = aligned_data['close_market'].pct_change()
                    
                    data['market_correlation'] = stock_returns.rolling(50).corr(market_returns)
                    
                    # Beta calculation
                    covariance = stock_returns.rolling(50).cov(market_returns)
                    market_variance = market_returns.rolling(50).var()
                    data['rolling_beta'] = covariance / market_variance
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in market correlation analysis: {str(e)}")
            return data
    
    def _add_options_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add options-based risk indicators (placeholder for future implementation)"""
        try:
            # This would integrate with options data providers
            # For now, we'll add placeholder columns
            data['put_call_ratio'] = np.nan
            data['implied_volatility'] = np.nan
            data['options_volume'] = np.nan
            
            # Future implementation would include:
            # - Put/Call ratio
            # - Implied volatility
            # - Options volume
            # - Gamma exposure
            # - Options flow sentiment
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in options indicators: {str(e)}")
            return data
    
    def _get_market_benchmark(self) -> Optional[pd.DataFrame]:
        """Get market benchmark data (SPY)"""
        try:
            cache_key = "SPY_benchmark_1y"
            
            if self._is_cache_valid(cache_key):
                return self._get_from_cache(cache_key)
            
            spy_data = self.providers[self.primary_provider].get_historical_data('SPY', '1y')
            if spy_data is not None:
                self._cache_data(cache_key, spy_data)
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"Error getting market benchmark: {str(e)}")
            return None
    
    def get_portfolio_risk_analysis(self, portfolio: Dict[str, float]) -> Dict:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            portfolio: Dict with symbol as key and weight as value
        """
        try:
            portfolio_data = {}
            correlation_matrix = pd.DataFrame()
            
            # Get data for all portfolio symbols
            for symbol, weight in portfolio.items():
                data = self.get_stock_data(symbol, period="1y")
                if data is not None:
                    portfolio_data[symbol] = {
                        'data': data,
                        'weight': weight,
                        'returns': data['close'].pct_change().dropna()
                    }
            
            if not portfolio_data:
                return {'error': 'No valid data for portfolio symbols'}
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame({
                symbol: info['returns'] for symbol, info in portfolio_data.items()
            })
            correlation_matrix = returns_df.corr()
            
            # Portfolio metrics
            weights = np.array([info['weight'] for info in portfolio_data.values()])
            returns_matrix = returns_df.values
            
            # Portfolio return and volatility
            portfolio_returns = np.dot(returns_matrix, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
            portfolio_return = np.mean(portfolio_returns) * 252
            
            # Value at Risk (VaR)
            var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
            var_99 = np.percentile(portfolio_returns, 1) * np.sqrt(252)
            
            # Expected Shortfall (CVaR)
            cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
            
            # Maximum Drawdown
            cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Diversification metrics
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            diversification_ratio = portfolio_volatility / np.average([
                portfolio_data[symbol]['returns'].std() * np.sqrt(252) 
                for symbol in portfolio_data.keys()
            ], weights=weights)
            
            # Risk contribution by asset
            marginal_contrib = {}
            for i, (symbol, info) in enumerate(portfolio_data.items()):
                # Marginal contribution to portfolio volatility
                marginal_contrib[symbol] = (weights[i] * np.dot(returns_df.cov() * 252, weights)[i]) / portfolio_volatility**2
            
            # Overall portfolio risk level
            risk_factors = 0
            if portfolio_volatility > 0.25: risk_factors += 2
            elif portfolio_volatility > 0.18: risk_factors += 1
            
            if max_drawdown < -0.2: risk_factors += 2
            elif max_drawdown < -0.15: risk_factors += 1
            
            if avg_correlation > 0.7: risk_factors += 2
            elif avg_correlation > 0.5: risk_factors += 1
            
            if var_95 < -0.15: risk_factors += 1
            
            if risk_factors >= 5:
                overall_risk = 'very_high'
            elif risk_factors >= 3:
                overall_risk = 'high'
            elif risk_factors >= 2:
                overall_risk = 'medium'
            else:
                overall_risk = 'low'
            
            return {
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'avg_correlation': avg_correlation,
                'diversification_ratio': diversification_ratio,
                'risk_contribution': marginal_contrib,
                'correlation_matrix': correlation_matrix.to_dict(),
                'overall_risk_level': overall_risk,
                'risk_factors_count': risk_factors,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio risk analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_trading_signals(self, symbol: str, signal_types: List[str] = None) -> Dict:
        """
        Generate comprehensive trading signals with risk assessment
        """
        if signal_types is None:
            signal_types = ['momentum', 'mean_reversion', 'breakout', 'volume', 'risk']
        
        try:
            data = self.get_stock_data(symbol, period="6mo")
            if data is None:
                return {'error': f'No data available for {symbol}'}
            
            signals = {'symbol': symbol, 'timestamp': datetime.now().isoformat(), 'signals': {}}
            
            # Momentum signals
            if 'momentum' in signal_types:
                signals['signals']['momentum'] = self._generate_momentum_signals(data)
            
            # Mean reversion signals
            if 'mean_reversion' in signal_types:
                signals['signals']['mean_reversion'] = self._generate_mean_reversion_signals(data)
            
            # Breakout signals
            if 'breakout' in signal_types:
                signals['signals']['breakout'] = self._generate_breakout_signals(data)
            
            # Volume signals
            if 'volume' in signal_types:
                signals['signals']['volume'] = self._generate_volume_signals(data)
            
            # Risk signals
            if 'risk' in signal_types:
                signals['signals']['risk'] = self._generate_risk_signals(data)
            
            # Composite signal
            signals['composite_signal'] = self._calculate_composite_signal(signals['signals'])
            
            # Store signals in database
            self._store_trading_signals(symbol, signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> Dict:
        """Generate momentum-based signals"""
        try:
            latest = data.iloc[-1]
            signals = {}
            
            # MACD signal
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_cross = (data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]) and \
                           (data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2])
                signals['macd_bullish'] = 1.0 if macd_cross else 0.0
            
            # RSI momentum
            if 'rsi' in data.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    signals['rsi_oversold'] = 1.0
                elif rsi > 70:
                    signals['rsi_overbought'] = -1.0
                else:
                    signals['rsi_neutral'] = 0.0
            
            # Moving average trend
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                ma_signal = 1.0 if latest['close'] > latest['sma_20'] > latest['sma_50'] else -1.0
                signals['ma_trend'] = ma_signal
            
            # Price momentum
            price_change_20d = (latest['close'] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            if price_change_20d > 0.05:
                signals['price_momentum'] = 1.0
            elif price_change_20d < -0.05:
                signals['price_momentum'] = -1.0
            else:
                signals['price_momentum'] = 0.0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in momentum signals: {str(e)}")
            return {}
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> Dict:
        """Generate mean reversion signals"""
        try:
            latest = data.iloc[-1]
            signals = {}
            
            # Bollinger Bands
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                if latest['close'] > latest['bb_upper']:
                    signals['bb_overbought'] = -1.0
                elif latest['close'] < latest['bb_lower']:
                    signals['bb_oversold'] = 1.0
                else:
                    signals['bb_neutral'] = 0.0
            
            # Z-score mean reversion
            price_mean = data['close'].rolling(50).mean().iloc[-1]
            price_std = data['close'].rolling(50).std().iloc[-1]
            z_score = (latest['close'] - price_mean) / price_std
            
            if z_score > 2:
                signals['z_score_reversion'] = -1.0
            elif z_score < -2:
                signals['z_score_reversion'] = 1.0
            else:
                signals['z_score_reversion'] = 0.0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion signals: {str(e)}")
            return {}
    
    def _generate_breakout_signals(self, data: pd.DataFrame) -> Dict:
        """Generate breakout signals"""
        try:
            latest = data.iloc[-1]
            signals = {}
            
            # Support/Resistance breakout
            if 'resistance' in data.columns and 'support' in data.columns:
                if latest['close'] > latest['resistance'] * 1.02:  # 2% above resistance
                    signals['resistance_breakout'] = 1.0
                elif latest['close'] < latest['support'] * 0.98:  # 2% below support
                    signals['support_breakdown'] = -1.0
                else:
                    signals['range_bound'] = 0.0
            
            # Volume-confirmed breakout
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if latest['volume'] > avg_volume * 1.5:  # 50% above average volume
                price_change = (latest['close'] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                if price_change > 0.03:  # 3% price increase with high volume
                    signals['volume_breakout'] = 1.0
                elif price_change < -0.03:
                    signals['volume_breakdown'] = -1.0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in breakout signals: {str(e)}")
            return {}
    
    def _generate_volume_signals(self, data: pd.DataFrame) -> Dict:
        """Generate volume-based signals"""
        try:
            latest = data.iloc[-1]
            signals = {}
            
            # Volume trend
            volume_ma_short = data['volume'].rolling(5).mean().iloc[-1]
            volume_ma_long = data['volume'].rolling(20).mean().iloc[-1]
            
            if volume_ma_short > volume_ma_long * 1.2:
                signals['volume_increasing'] = 1.0
            elif volume_ma_short < volume_ma_long * 0.8:
                signals['volume_decreasing'] = -1.0
            else:
                signals['volume_stable'] = 0.0
            
            # Price-Volume relationship
            price_change = (latest['close'] - data['close'].iloc[-2]) / data['close'].iloc[-2]
            volume_ratio = latest['volume'] / data['volume'].rolling(20).mean().iloc[-1]
            
            if price_change > 0 and volume_ratio > 1.5:
                signals['bullish_volume_confirmation'] = 1.0
            elif price_change < 0 and volume_ratio > 1.5:
                signals['bearish_volume_confirmation'] = -1.0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in volume signals: {str(e)}")
            return {}
    
    def _generate_risk_signals(self, data: pd.DataFrame) -> Dict:
        """Generate risk-based signals"""
        try:
            signals = {}
            
            # Get risk metrics from data attributes
            if hasattr(data, 'attrs') and 'risk_metrics' in data.attrs:
                risk_metrics = data.attrs['risk_metrics']
                
                risk_level = risk_metrics.get('risk_level', 'medium')
                volatility = risk_metrics.get('volatility', 0.2)
                
                # Risk-adjusted position sizing suggestion
                if risk_level == 'low' and volatility < 0.15:
                    signals['position_size_multiplier'] = 1.0
                elif risk_level == 'medium' or volatility < 0.25:
                    signals['position_size_multiplier'] = 0.7
                else:
                    signals['position_size_multiplier'] = 0.3
                
                # Risk warning signals
                if volatility > 0.4:
                    signals['high_volatility_warning'] = -1.0
                
                if risk_metrics.get('max_drawdown', 0) < -0.3:
                    signals['high_drawdown_warning'] = -1.0
            
            # Market regime risk
            if 'market_regime' in data.columns:
                latest_regime = data['market_regime'].iloc[-1]
                if latest_regime == 'bear':
                    signals['bear_market_caution'] = -0.5
                elif latest_regime == 'bull':
                    signals['bull_market_confidence'] = 0.5
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in risk signals: {str(e)}")
            return {}
    
    def _calculate_composite_signal(self, signals: Dict) -> Dict:
        """Calculate composite trading signal from all individual signals"""
        try:
            all_signals = []
            signal_weights = {
                'momentum': 0.3,
                'mean_reversion': 0.2,
                'breakout': 0.25,
                'volume': 0.15,
                'risk': 0.1
            }
            
            composite_score = 0.0
            total_weight = 0.0
            
            for signal_type, weight in signal_weights.items():
                if signal_type in signals:
                    type_signals = signals[signal_type]
                    if type_signals:
                        # Average signals within each type
                        avg_signal = np.mean([v for v in type_signals.values() if isinstance(v, (int, float))])
                        composite_score += avg_signal * weight
                        total_weight += weight
            
            if total_weight > 0:
                composite_score = composite_score / total_weight
            
            # Classify signal strength
            if composite_score > 0.3:
                signal_strength = 'strong_buy'
            elif composite_score > 0.1:
                signal_strength = 'buy'
            elif composite_score > -0.1:
                signal_strength = 'hold'
            elif composite_score > -0.3:
                signal_strength = 'sell'
            else:
                signal_strength = 'strong_sell'
            
            return {
                'composite_score': composite_score,
                'signal_strength': signal_strength,
                'confidence': min(abs(composite_score) * 2, 1.0)  # Confidence based on signal strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {str(e)}")
            return {'composite_score': 0.0, 'signal_strength': 'hold', 'confidence': 0.0}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        return self.data_cache.get(cache_key)
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with expiry"""
        with self.data_lock:
            self.data_cache[cache_key] = data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
    
    def _update_risk_database(self, symbol: str, data: pd.DataFrame):
        """Update risk metrics in database"""
        try:
            if hasattr(data, 'attrs') and 'risk_metrics' in data.attrs:
                risk_metrics = data.attrs['risk_metrics']
                
                conn = sqlite3.connect(self.cache_db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO risk_metrics 
                    (symbol, risk_level, volatility, beta, max_drawdown, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    risk_metrics.get('risk_level', 'unknown'),
                    risk_metrics.get('volatility', 0.0),
                    risk_metrics.get('beta', 1.0),
                    risk_metrics.get('max_drawdown', 0.0),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Error updating risk database for {symbol}: {str(e)}")
    
    def _store_trading_signals(self, symbol: str, signals: Dict):
        """Store trading signals in database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            composite = signals.get('composite_signal', {})
            
            cursor.execute('''
                INSERT INTO trading_signals 
                (symbol, signal_type, strength, timestamp, conditions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                composite.get('signal_strength', 'hold'),
                composite.get('composite_score', 0.0),
                datetime.now().isoformat(),
                json.dumps(signals['signals'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing signals for {symbol}: {str(e)}")
    
    def get_provider_status(self) -> Dict:
        """Get status of all data providers"""
        status = {}
        
        for name, provider in self.providers.items():
            provider_health = provider.health_check()
            performance = self.provider_performance[name]
            
            status[name] = {
                'health': provider_health,
                'performance': performance,
                'is_primary': name == self.primary_provider
            }
        
        return status
    
    def switch_primary_provider(self, provider_name: str):
        """Switch primary data provider"""
        if provider_name in self.providers:
            old_primary = self.primary_provider
            self.primary_provider = provider_name
            self.backup_provider = old_primary
            self.logger.info(f"Switched primary provider to {provider_name}")
        else:
            self.logger.error(f"Provider {provider_name} not found")
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.data_lock:
            # Clean memory cache
            expired_keys = [
                key for key, expiry in self.cache_expiry.items() 
                if expiry < cutoff_time
            ]
            
            for key in expired_keys:
                self.data_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
            
            self.logger.info(f"Cleaned up {len(expired_keys)} cache entries")
        
        # Clean database cache
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM price_data WHERE datetime(expiry) < datetime(?)",
                (cutoff_time.isoformat(),)
            )
            
            cursor.execute(
                "DELETE FROM trading_signals WHERE datetime(timestamp) < datetime(?)",
                (cutoff_time.isoformat(),)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning database cache: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
            # GARCH-like volatility estimation
            returns_squared = returns ** 2
            vol_short = returns_squared.rolling(5).mean()
            vol_long = returns_squared.rolling(20).mean()
            
            data['volatility_ratio'] = vol_short / vol_long
            data['volatility_regime'] = np.where(
                data['volatility_ratio'] > 1.5, 'high_vol',
                np.where(data['volatility_ratio'] < 0.5, 'low_vol', 'normal_vol')
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in volatility clustering: {str(e)}")
            return data
    
    def _add_support_resistance_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify support and resistance levels"""
        try:
            window = 20
            
            # Rolling max/min for resistance/support
            data['resistance'] = data['high'].rolling(window).max()
            data['support'] = data['low'].rolling(window).min()
            
            # Distance from support/resistance
            data['dist_from_resistance'] = (data['resistance'] - data['close']) / data['close']
            data['dist_from_support'] = (data['close'] - data['support']) / data['close']
            
            # Risk signals
            data['near_resistance'] = data['dist_from_resistance'] < 0.02  # Within 2%
            data['near_support'] = data['dist_from_support'] < 0.02
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in support/resistance analysis: {str(e)}")
            return data
    
    def _add_risk_adjusted_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted momentum indicators"""
        try:
            returns = data['close'].pct_change()
            