import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .provider_base import ProviderBase
import json
import time

class AlphaVantageProvider(ProviderBase):
    """
    Alpha Vantage data provider with premium features and risk management
    """
    
    def __init__(self, api_key: str = "demo"):
        super().__init__("AlphaVantage", rate_limit=5)  # 5 requests per minute for free tier
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_minute_requests = []
        
        if api_key == "demo":
            self.logger.warning("Using demo API key - limited functionality")
    
    def _check_rate_limit(self):
        """Enhanced rate limiting for Alpha Vantage API"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.last_minute_requests = [
            req_time for req_time in self.last_minute_requests 
            if current_time - req_time < 60
        ]
        
        # Check if we've exceeded the rate limit
        if len(self.last_minute_requests) >= 5:  # 5 requests per minute
            sleep_time = 60 - (current_time - self.last_minute_requests[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
        
        self.last_minute_requests.append(current_time)
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with error handling"""
        self._check_rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                self.logger.error(f"API Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                self.logger.warning(f"API Note: {data['Note']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e, str(params))
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical data with technical indicators
        """
        # Map period to Alpha Vantage function
        if period in ["1d", "5d"]:
            function = "TIME_SERIES_INTRADAY"
            interval = "5min"
            time_key = "Time Series (5min)"
        else:
            function = "TIME_SERIES_DAILY_ADJUSTED"
            time_key = "Time Series (Daily)"
        
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': 'full'
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params['interval'] = interval
        
        try:
            data = self._make_request(params)
            if not data:
                return None
            
            # Extract time series data
            time_series = data.get(time_key, {})
            if not time_series:
                self.logger.error(f"No time series data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '5. volume': 'volume',
                '6. volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            # Use adjusted close if available, otherwise use close
            if 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            
            # Filter by period if needed
            if period != "max":
                days_map = {
                    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
                }
                if period in days_map:
                    cutoff_date = datetime.now() - timedelta(days=days_map[period])
                    df = df[df.index >= cutoff_date]
            
            if not self._validate_data(df, symbol):
                return None
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Add Alpha Vantage specific indicators
            df = self._add_alpha_vantage_indicators(df, symbol)
            
            # Add risk metrics
            risk_metrics = self._calculate_risk_metrics(df)
            df.attrs['risk_metrics'] = risk_metrics
            df.attrs['symbol'] = symbol
            df.attrs['provider'] = self.name
            df.attrs['last_updated'] = datetime.now().isoformat()
            
            self.connection_failures = 0
            self.logger.info(f"Retrieved {len(df)} records for {symbol} from Alpha Vantage")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding Alpha Vantage indicators for {symbol}: {str(e)}")
            return df
    
    def _get_sma_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get Simple Moving Average indicators"""
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': 20,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        if not data or 'Technical Analysis: SMA' not in data:
            return None
        
        sma_data = data['Technical Analysis: SMA']
        df = pd.DataFrame.from_dict(sma_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['sma_20_av']
        df = df.astype(float)
        
        return df
    
    def _get_ema_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get Exponential Moving Average indicators"""
        params = {
            'function': 'EMA',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': 12,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        if not data or 'Technical Analysis: EMA' not in data:
            return None
        
        ema_data = data['Technical Analysis: EMA']
        df = pd.DataFrame.from_dict(ema_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['ema_12_av']
        df = df.astype(float)
        
        return df
    
    def _get_rsi(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get RSI indicator"""
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        if not data or 'Technical Analysis: RSI' not in data:
            return None
        
        rsi_data = data['Technical Analysis: RSI']
        df = pd.DataFrame.from_dict(rsi_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['rsi_av']
        df = df.astype(float)
        
        return df
    
    def _get_macd(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get MACD indicator"""
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': 'daily',
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        if not data or 'Technical Analysis: MACD' not in data:
            return None
        
        macd_data = data['Technical Analysis: MACD']
        df = pd.DataFrame.from_dict(macd_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['macd_av', 'macd_hist_av', 'macd_signal_av']
        df = df.astype(float)
        
        return df
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote data"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        try:
            data = self._make_request(params)
            if not data or 'Global Quote' not in data:
                return None
            
            quote_data = data['Global Quote']
            
            real_time_data = {
                'symbol': quote_data.get('01. symbol', symbol),
                'current_price': float(quote_data.get('05. price', 0)),
                'previous_close': float(quote_data.get('08. previous close', 0)),
                'open': float(quote_data.get('02. open', 0)),
                'day_high': float(quote_data.get('03. high', 0)),
                'day_low': float(quote_data.get('04. low', 0)),
                'volume': int(quote_data.get('06. volume', 0)),
                'latest_trading_day': quote_data.get('07. latest trading day', ''),
                'price_change': float(quote_data.get('09. change', 0)),
                'price_change_percent': quote_data.get('10. change percent', '0%').rstrip('%'),
                'timestamp': datetime.now().isoformat(),
                'provider': self.name
            }
            
            # Convert percentage string to float
            try:
                real_time_data['price_change_percent'] = float(real_time_data['price_change_percent'])
            except:
                real_time_data['price_change_percent'] = 0.0
            
            # Risk assessment
            daily_volatility = abs(real_time_data['price_change_percent'])
            if daily_volatility > 5:
                real_time_data['risk_alert'] = 'high_volatility'
            elif daily_volatility > 3:
                real_time_data['risk_alert'] = 'medium_volatility'
            else:
                real_time_data['risk_alert'] = 'normal'
            
            # Get additional risk data
            risk_data = self._get_risk_metrics(symbol)
            if risk_data:
                real_time_data.update(risk_data)
            
            self.connection_failures = 0
            return real_time_data
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None
    
    def _get_risk_metrics(self, symbol: str) -> Optional[Dict]:
        """Get additional risk metrics from Alpha Vantage"""
        try:
            # Get company overview for fundamental analysis
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol
            }
            
            data = self._make_request(params)
            if not data:
                return None
            
            risk_metrics = {
                'beta': float(data.get('Beta', 1.0)) if data.get('Beta', 'None') != 'None' else 1.0,
                'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio', 'None') != 'None' else 0,
                'peg_ratio': float(data.get('PEGRatio', 0)) if data.get('PEGRatio', 'None') != 'None' else 0,
                'dividend_yield': float(data.get('DividendYield', 0)) if data.get('DividendYield', 'None') != 'None' else 0,
                'profit_margin': float(data.get('ProfitMargin', 0)) if data.get('ProfitMargin', 'None') != 'None' else 0,
                'debt_to_equity': float(data.get('DebtToEquityRatio', 0)) if data.get('DebtToEquityRatio', 'None') != 'None' else 0,
                'return_on_equity': float(data.get('ReturnOnEquityTTM', 0)) if data.get('ReturnOnEquityTTM', 'None') != 'None' else 0,
                'analyst_target_price': float(data.get('AnalystTargetPrice', 0)) if data.get('AnalystTargetPrice', 'None') != 'None' else 0,
                'fifty_two_week_high': float(data.get('52WeekHigh', 0)) if data.get('52WeekHigh', 'None') != 'None' else 0,
                'fifty_two_week_low': float(data.get('52WeekLow', 0)) if data.get('52WeekLow', 'None') != 'None' else 0,
                'market_cap': int(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization', 'None') != 'None' else 0
            }
            
            # Calculate composite risk score
            risk_metrics['fundamental_risk_score'] = self._calculate_alpha_vantage_risk_score(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics for {symbol}: {str(e)}")
            return None
    
    def _calculate_alpha_vantage_risk_score(self, metrics: Dict) -> str:
        """Calculate risk score from Alpha Vantage fundamental data"""
        risk_score = 0
        
        # Beta risk
        beta = metrics.get('beta', 1.0)
        if beta > 1.5:
            risk_score += 2
        elif beta > 1.2:
            risk_score += 1
        
        # PE ratio risk
        pe_ratio = metrics.get('pe_ratio', 0)
        if pe_ratio > 50 or pe_ratio < 0:
            risk_score += 2
        elif pe_ratio > 25:
            risk_score += 1
        
        # Debt to equity risk
        debt_to_equity = metrics.get('debt_to_equity', 0)
        if debt_to_equity > 1.0:
            risk_score += 2
        elif debt_to_equity > 0.5:
            risk_score += 1
        
        # Market cap risk
        market_cap = metrics.get('market_cap', 0)
        if market_cap < 1e9:  # Less than $1B
            risk_score += 2
        elif market_cap < 10e9:  # Less than $10B
            risk_score += 1
        
        # Profit margin risk
        profit_margin = metrics.get('profit_margin', 0)
        if profit_margin < 0:  # Negative margins
            risk_score += 2
        elif profit_margin < 0.05:  # Less than 5%
            risk_score += 1
        
        if risk_score >= 6:
            return 'very_high'
        elif risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_market_status(self) -> Dict:
        """Get market status using Alpha Vantage data"""
        try:
            # Get major indices
            indices = ['SPY', 'QQQ', 'DIA']  # ETFs representing major indices
            market_data = {}
            
            for index in indices:
                quote_data = self.get_real_time_data(index)
                if quote_data:
                    market_data[index] = {
                        'current_price': quote_data['current_price'],
                        'change_percent': quote_data['price_change_percent'],
                        'trend': 'up' if quote_data['price_change_percent'] > 0 else 'down' if quote_data['price_change_percent'] < 0 else 'flat'
                    }
            
            # Determine market sentiment
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
            
            return {
                'market_sentiment': market_sentiment,
                'indices': market_data,
                'timestamp': datetime.now().isoformat(),
                'provider': self.name,
                'risk_level': self._assess_market_risk_av(market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {
                'market_sentiment': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'provider': self.name,
                'error': str(e)
            }
    
    def _assess_market_risk_av(self, market_data: Dict) -> str:
        """Assess market risk based on Alpha Vantage data"""
        if not market_data:
            return 'unknown'
        
        changes = [abs(data['change_percent']) for data in market_data.values()]
        avg_volatility = np.mean(changes)
        
        if avg_volatility > 3.0:
            return 'very_high'
        elif avg_volatility > 2.0:
            return 'high'
        elif avg_volatility > 1.0:
            return 'medium' 
        else:
            return 'low'
    
    def get_earnings_data(self, symbol: str) -> Optional[Dict]:
        """Get earnings data for fundamental analysis"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol
        }
        
        try:
            data = self._make_request(params)
            if not data:
                return None
            
            annual_earnings = data.get('annualEarnings', [])
            quarterly_earnings = data.get('quarterlyEarnings', [])
            
            earnings_data = {
                'symbol': symbol,
                'annual_earnings': annual_earnings,
                'quarterly_earnings': quarterly_earnings,
                'timestamp': datetime.now().isoformat(),
                'provider': self.name
            }
            
            # Calculate earnings growth risk
            if len(quarterly_earnings) >= 4:
                recent_quarters = quarterly_earnings[:4]
                earnings_growth = []
                
                for i in range(1, len(recent_quarters)):
                    try:
                        current = float(recent_quarters[i-1]['reportedEPS'])
                        previous = float(recent_quarters[i]['reportedEPS'])
                        if previous != 0:
                            growth = (current - previous) / abs(previous)
                            earnings_growth.append(growth)
                    except:
                        continue
                
                if earnings_growth:
                    avg_growth = np.mean(earnings_growth)
                    growth_volatility = np.std(earnings_growth)
                    
                    earnings_data['avg_earnings_growth'] = avg_growth
                    earnings_data['earnings_volatility'] = growth_volatility
                    
                    # Risk assessment
                    if growth_volatility > 0.5 or avg_growth < -0.2:
                        earnings_data['earnings_risk'] = 'high'
                    elif growth_volatility > 0.3 or avg_growth < 0:
                        earnings_data['earnings_risk'] = 'medium'
                    else:
                        earnings_data['earnings_risk'] = 'low'
            
            return earnings_data
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None
    
    def get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get news sentiment analysis"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'limit': 50
        }
        
        try:
            data = self._make_request(params)
            if not data or 'feed' not in data:
                return None
            
            news_items = data['feed']
            
            # Analyze sentiment
            sentiment_scores = []
            relevance_scores = []
            
            for item in news_items:
                ticker_sentiments = item.get('ticker_sentiment', [])
                for ticker_data in ticker_sentiments:
                    if ticker_data.get('ticker') == symbol:
                        try:
                            sentiment_scores.append(float(ticker_data.get('ticker_sentiment_score', 0)))
                            relevance_scores.append(float(ticker_data.get('relevance_score', 0)))
                        except:
                            continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_volatility = np.std(sentiment_scores)
                avg_relevance = np.mean(relevance_scores)
                
                # Classify sentiment
                if avg_sentiment > 0.1:
                    sentiment_label = 'positive'
                elif avg_sentiment < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                # Risk from sentiment volatility
                if sentiment_volatility > 0.3:
                    sentiment_risk = 'high'
                elif sentiment_volatility > 0.2:
                    sentiment_risk = 'medium'
                else:
                    sentiment_risk = 'low'
                
                return {
                    'symbol': symbol,
                    'avg_sentiment_score': avg_sentiment,
                    'sentiment_label': sentiment_label,
                    'sentiment_volatility': sentiment_volatility,
                    'sentiment_risk': sentiment_risk,
                    'avg_relevance': avg_relevance,
                    'news_count': len(sentiment_scores),
                    'timestamp': datetime.now().isoformat(),
                    'provider': self.name
                }
            
            return None
            
        except Exception as e:
            self._handle_request_error(e, symbol)
            return None_handle_request_error(e, symbol)
            return None
    
    def _add_alpha_vantage_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add Alpha Vantage specific technical indicators"""
        try:
            # Get SMA indicators
            sma_indicators = self._get_sma_indicators(symbol)
            if sma_indicators:
                df = pd.merge(df, sma_indicators, left_index=True, right_index=True, how='left')
            
            # Get EMA indicators  
            ema_indicators = self._get_ema_indicators(symbol)
            if ema_indicators:
                df = pd.merge(df, ema_indicators, left_index=True, right_index=True, how='left')
            
            # Get RSI
            rsi_data = self._get_rsi(symbol)
            if rsi_data:
                df = pd.merge(df, rsi_data, left_index=True, right_index=True, how='left')
            
            # Get MACD
            macd_data = self._get_macd(symbol)
            if macd_data:
                df = pd.merge(df, macd_data, left_index=True, right_index=True, how='left')
            
            return df
            
        except Exception as e:
            self.