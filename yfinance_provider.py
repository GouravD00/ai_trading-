import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .provider_base import ProviderBase
import requests

class YFinanceProvider(ProviderBase):
    """
    Yahoo Finance data provider with enhanced error handling and risk management
    """
    
    def __init__(self):
        super().__init__("YFinance", rate_limit=10)  # 10 requests per second
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical data with comprehensive risk analysis
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        self._rate_limit_check()
        
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            data = ticker.history(period=period, auto_adjust=True, prepost=True)
            
            if not self._validate_data(data, symbol):
                return None
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add risk metrics as metadata
            risk_metrics = self._calculate_risk_metrics(data)
            data.attrs['risk_metrics'] = risk_metrics
            data.attrs['symbol'] = symbol
            data.attrs['provider'] = self.name
            data.attrs['last_updated'] = datetime.now().isoformat()
            
            # Reset connection failures on success
            self.connection_failures = 0
            
            self.logger.info(f"Successfully retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote data with risk assessment"""
        self._rate_limit_check()
        
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            
            # Get fast info for real-time data
            fast_info = ticker.fast_info
            
            current_price = fast_info.get('lastPrice', info.get('currentPrice', 0))
            
            if current_price == 0:
                self.logger.warning(f"No current price available for {symbol}")
                return None
            
            # Get recent data for short-term analysis
            recent_data = self.get_historical_data(symbol, "5d")
            
            real_time_data = {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': fast_info.get('previousClose', info.get('previousClose', 0)),
                'open': fast_info.get('open', info.get('open', 0)),
                'day_high': fast_info.get('dayHigh', info.get('dayHigh', 0)),
                'day_low': fast_info.get('dayLow', info.get('dayLow', 0)),
                'volume': fast_info.get('lastVolume', info.get('volume', 0)),
                'market_cap': fast_info.get('marketCap', info.get('marketCap', 0)),
                'timestamp': datetime.now().isoformat(),
                'provider': self.name
            }
            
            # Calculate intraday metrics
            if real_time_data['previous_close'] > 0:
                price_change = current_price - real_time_data['previous_close']
                real_time_data['price_change'] = price_change
                real_time_data['price_change_percent'] = (price_change / real_time_data['previous_close']) * 100
                
                # Risk assessment based on intraday movement
                daily_volatility = abs(real_time_data['price_change_percent'])
                if daily_volatility > 5:
                    real_time_data['risk_alert'] = 'high_volatility'
                elif daily_volatility > 3:
                    real_time_data['risk_alert'] = 'medium_volatility'
                else:
                    real_time_data['risk_alert'] = 'normal'
            
            # Add 5-day risk metrics if available
            if recent_data is not None and 'risk_metrics' in recent_data.attrs:
                real_time_data['risk_metrics'] = recent_data.attrs['risk_metrics']
            
            # Trading session status
            real_time_data['trading_session'] = self._get_trading_session_status()
            
            self.connection_failures = 0
            return real_time_data
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None
    
    def get_market_status(self) -> Dict:
        """Get overall market status"""
        try:
            # Use major market indices to determine market status
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
            market_data = {}
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index, session=self.session)
                    fast_info = ticker.fast_info
                    
                    current_price = fast_info.get('lastPrice', 0)
                    previous_close = fast_info.get('previousClose', 0)
                    
                    if current_price > 0 and previous_close > 0:
                        change_percent = ((current_price - previous_close) / previous_close) * 100
                        market_data[index] = {
                            'current_price': current_price,
                            'change_percent': change_percent,
                            'trend': 'up' if change_percent > 0 else 'down' if change_percent < 0 else 'flat'
                        }
                except:
                    continue
            
            # Determine overall market sentiment
            if market_data:
                avg_change = np.mean([data['change_percent'] for data in market_data.values()])
                if avg_change > 1:
                    market_sentiment = 'bullish'
                elif avg_change < -1:
                    market_sentiment = 'bearish'
                else:
                    market_sentiment = 'neutral'
            else:
                market_sentiment = 'unknown'
            
            trading_session = self._get_trading_session_status()
            
            return {
                'market_sentiment': market_sentiment,
                'trading_session': trading_session,
                'indices': market_data,
                'timestamp': datetime.now().isoformat(),
                'provider': self.name,
                'risk_level': self._assess_market_risk(market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {
                'market_sentiment': 'unknown',
                'trading_session': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'provider': self.name,
                'error': str(e)
            }
    
    def _get_trading_session_status(self) -> str:
        """Determine current trading session status"""
        now = datetime.now()
        
        # US market hours (EST/EDT)
        if now.weekday() >= 5:  # Weekend
            return 'closed'
        
        # Simple hour check (would need timezone handling for production)
        current_hour = now.hour
        if 9 <= current_hour < 16:  # 9 AM to 4 PM EST (simplified)
            return 'open'
        elif 4 <= current_hour < 9:  # Pre-market
            return 'pre_market'
        elif 16 <= current_hour < 20:  # After-hours
            return 'after_hours'
        else:
            return 'closed'
    
    def _assess_market_risk(self, market_data: Dict) -> str:
        """Assess overall market risk based on index movements"""
        if not market_data:
            return 'unknown'
        
        changes = [abs(data['change_percent']) for data in market_data.values()]
        avg_volatility = np.mean(changes)
        
        if avg_volatility > 2.5:
            return 'high'
        elif avg_volatility > 1.5:
            return 'medium'
        else:
            return 'low'
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive company information for risk assessment"""
        self._rate_limit_check()
        
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            
            # Extract key risk-relevant information
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'beta': info.get('beta', 1.0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margins': info.get('profitMargins', 0),
                'analyst_recommendation': info.get('recommendationMean', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate fundamental risk score
            company_info['fundamental_risk_score'] = self._calculate_fundamental_risk(company_info)
            
            return company_info
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None
    
    def _calculate_fundamental_risk(self, info: Dict) -> str:
        """Calculate fundamental risk based on company metrics"""
        risk_score = 0
        
        # Beta risk (market sensitivity)
        beta = info.get('beta', 1.0)
        if beta > 1.5:
            risk_score += 2
        elif beta > 1.2:
            risk_score += 1
        
        # Debt to equity risk
        debt_to_equity = info.get('debt_to_equity', 0)
        if debt_to_equity > 1.0:
            risk_score += 2
        elif debt_to_equity > 0.5:
            risk_score += 1
        
        # PE ratio risk
        pe_ratio = info.get('pe_ratio', 0)
        if pe_ratio > 50 or pe_ratio < 0:
            risk_score += 2
        elif pe_ratio > 25:
            risk_score += 1
        
        # Market cap risk (smaller companies = higher risk)
        market_cap = info.get('market_cap', 0)
        if market_cap < 1e9:  # Less than $1B
            risk_score += 2
        elif market_cap < 10e9:  # Less than $10B
            risk_score += 1
        
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'