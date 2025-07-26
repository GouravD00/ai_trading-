import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from datahub import DataManager, DEFAULT_CONFIG

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    position_value: float
    risk_amount: float
    position_type: str  # 'long' or 'short'
    
class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config['risk_management']
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        
        # Risk limits
        self.max_position_size = self.config.get('max_position_size', 0.02)  # 2% max per position
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.10)  # 10% total portfolio risk
        self.max_daily_loss = self.config.get('max_daily_loss', 0.03)  # 3% max daily loss
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.correlation_limit = self.config.get('correlation_limit', 0.7)  # Max correlation between positions
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.risk_budget_used = 0.0
        
    def validate_new_position(self, symbol: str, signal_strength: float, 
                            current_positions: Dict, portfolio_value: float,
                            stock_data: pd.DataFrame) -> Dict:
        """
        Comprehensive position validation with multiple risk checks
        """
        validation_result = {
            'approved': False,
            'position_size': 0.0,
            'risk_amount': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'warnings': [],
            'rejection_reasons': []
        }
        
        try:
            # Check daily loss limit
            if self.daily_loss_limit_hit:
                validation_result['rejection_reasons'].append('Daily loss limit reached')
                return validation_result
            
            # Check maximum positions limit
            if len(current_positions) >= self.max_open_positions:
                validation_result['rejection_reasons'].append(f'Maximum positions limit ({self.max_open_positions}) reached')
                return validation_result
            
            # Get current price and risk metrics
            current_price = stock_data['close'].iloc[-1]
            
            # Risk metrics from data
            risk_metrics = getattr(stock_data, 'attrs', {}).get('risk_metrics', {})
            volatility = risk_metrics.get('volatility', 0.2)
            risk_level = risk_metrics.get('risk_level', 'medium')
            
            # Adjust position size based on volatility and signal strength
            base_position_size = self.max_position_size
            
            # Volatility adjustment
            if volatility > 0.4:
                base_position_size *= 0.5  # Reduce by 50% for high volatility
                validation_result['warnings'].append('High volatility detected - position size reduced')
            elif volatility > 0.25:
                base_position_size *= 0.75  # Reduce by 25% for medium-high volatility
            
            # Signal strength adjustment
            base_position_size *= min(abs(signal_strength), 1.0)
            
            # Risk level adjustment
            risk_multipliers = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'very_high': 0.3}
            base_position_size *= risk_multipliers.get(risk_level, 0.5)
            
            # Calculate position value and risk amount
            position_value = portfolio_value * base_position_size
            
            # ATR-based stop loss (more dynamic)
            atr = stock_data.get('atr', pd.Series([current_price * 0.02])).iloc[-1]  # Default 2% if no ATR
            stop_loss_distance = max(atr * 2, current_price * 0.01)  # At least 1% stop loss
            
            if signal_strength > 0:  # Long position
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + (stop_loss_distance * 2)  # 2:1 reward/risk
            else:  # Short position
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - (stop_loss_distance * 2)
            
            # Calculate risk amount
            risk_per_share = abs(current_price - stop_loss)
            max_shares = position_value / current_price
            risk_amount = max_shares * risk_per_share
            
            # Check if risk amount exceeds limits
            if risk_amount > portfolio_value * self.max_position_size:
                # Adjust position size to meet risk limit
                max_shares = (portfolio_value * self.max_position_size) / risk_per_share
                position_value = max_shares * current_price
                risk_amount = portfolio_value * self.max_position_size
                validation_result['warnings'].append('Position size reduced to meet risk limits')
            
            # Check portfolio-wide risk
            total_portfolio_risk = self.risk_budget_used + (risk_amount / portfolio_value)
            if total_portfolio_risk > self.max_portfolio_risk:
                validation_result['rejection_reasons'].append(f'Portfolio risk limit exceeded ({total_portfolio_risk:.2%} > {self.max_portfolio_risk:.2%})')
                return validation_result
            
            # Check correlation with existing positions
            correlation_warning = self._check_correlation_risk(symbol, current_positions, stock_data)
            if correlation_warning:
                validation_result['warnings'].append(correlation_warning)
            
            # Check liquidity
            avg_volume = stock_data['volume'].rolling(20).mean().iloc[-1]
            position_volume = max_shares
            if position_volume > avg_volume * 0.01:  # More than 1% of average daily volume
                validation_result['warnings'].append('Large position relative to average volume - execution risk')
            
            # Check market conditions
            if 'market_regime' in stock_data.columns:
                market_regime = stock_data['market_regime'].iloc[-1]
                if market_regime == 'bear' and signal_strength > 0:
                    base_position_size *= 0.7  # Reduce long positions in bear market
                    validation_result['warnings'].append('Bear market detected - long position size reduced')
            
            # Final validation
            if position_value < 100:  # Minimum position size
                validation_result['rejection_reasons'].append('Position size too small (minimum $100)')
                return validation_result
            
            # All checks passed
            validation_result.update({
                'approved': True,
                'position_size': position_value / portfolio_value,
                'position_value': position_value,
                'shares': max_shares,
                'risk_amount': risk_amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss)
            })
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in position validation for {symbol}: {str(e)}")
            validation_result['rejection_reasons'].append(f'Validation error: {str(e)}')
            return validation_result
    
    def _check_correlation_risk(self, symbol: str, current_positions: Dict, stock_data: pd.DataFrame) -> Optional[str]:
        """Check correlation risk with existing positions"""
        try:
            if not current_positions:
                return None
            
            # This would require correlation analysis between symbols
            # For now, we'll do a simple sector/industry check
            stock_attrs = getattr(stock_data, 'attrs', {})
            
            # Simple sector concentration check
            same_sector_count = 0
            for pos_symbol in current_positions.keys():
                # In real implementation, would check actual sector data
                same_sector_count += 1  # Placeholder
            
            if same_sector_count >= 3:
                return f'High concentration risk - {same_sector_count} positions in similar assets'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {str(e)}")
            return None
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L and check limits"""
        self.daily_pnl += pnl_change
        
        if self.daily_pnl <= -self.max_daily_loss:
            self.daily_loss_limit_hit = True
            self.logger.warning(f"Daily loss limit hit: {self.daily_pnl:.2%}")
    
    def reset_daily_limits(self):
        """Reset daily limits (call at start of each trading day)"""
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.logger.info("Daily risk limits reset")

class AITrader:
    """
    Main AI Trading System with advanced risk management and decision making
    """
    
    def __init__(self, config: Dict = None, alpha_vantage_key: str = "demo"):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = DataManager(alpha_vantage_key)
        self.risk_manager = RiskManager(self.config)
        
        # Portfolio and position tracking
        self.portfolio_value = 100000.0  # Starting with $100,000
        self.cash_balance = self.portfolio_value
        self.positions = {}  # Dict[str, Position]
        self.watchlist = set()
        
        # Trading state
        self.is_trading_enabled = True
        self.trading_session_active = False
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Database for persistence
        self.db_path = "ai_trader.db"
        self._init_database()
        
        self.logger.info("AI Trader initialized with enhanced risk management")
    
    def _init_database(self):
        """Initialize trading database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    quantity REAL,
                    entry_price REAL,
                    entry_time TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    position_type TEXT,
                    status TEXT,
                    pnl REAL,
                    risk_amount REAL
                )
            ''')
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    timestamp TEXT,
                    signal_strength REAL,
                    conditions TEXT
                )
            ''')
            
            # Portfolio history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    timestamp TEXT PRIMARY KEY,
                    portfolio_value REAL,
                    cash_balance REAL,
                    num_positions INTEGER,
                    daily_pnl REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
    
    def add_to_watchlist(self, symbols: List[str]):
        """Add symbols to watchlist for monitoring"""
        for symbol in symbols:
            symbol = symbol.upper()
            self.watchlist.add(symbol)
            self.logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbols: List[str]):
        """Remove symbols from watchlist"""
        for symbol in symbols:
            symbol = symbol.upper()
            self.watchlist.discard(symbol)
            self.logger.info(f"Removed {symbol} from watchlist")
    
    def scan_opportunities(self) -> List[Dict]:
        """
        Scan watchlist for trading opportunities
        """
        opportunities = []
        
        if not self.watchlist:
            self.logger.warning("Watchlist is empty")
            return opportunities
        
        for symbol in self.watchlist:
            try:
                # Skip if already have position
                if symbol in self.positions:
                    continue
                
                # Get trading signals
                signals = self.data_manager.generate_trading_signals(symbol)
                
                if 'error' in signals:
                    self.logger.warning(f"Error getting signals for {symbol}: {signals['error']}")
                    continue
                
                composite = signals.get('composite_signal', {})
                signal_strength = composite.get('composite_score', 0.0)
                confidence = composite.get('confidence', 0.0)
                
                # Filter for strong signals with high confidence
                if abs(signal_strength) > 0.3 and confidence > 0.6:
                    opportunities.append({
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'confidence': confidence,
                        'signal_type': composite.get('signal_strength', 'hold'),
                        'signals': signals['signals'],
                        'timestamp': signals['timestamp']
                    })
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {str(e)}")
                continue
        
        # Sort by signal strength and confidence
        opportunities.sort(key=lambda x: abs(x['signal_strength']) * x['confidence'], reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} trading opportunities")
        return opportunities
    
    def execute_trade(self, symbol: str, signal_data: Dict) -> Dict:
        """
        Execute a trade based on signal data with comprehensive risk management
        """
        try:
            signal_strength = signal_data['signal_strength']
            
            # Get stock data for risk analysis
            stock_data = self.data_manager.get_stock_data(symbol, period="6mo")
            if stock_data is None:
                return {'success': False, 'error': f'No data available for {symbol}'}
            
            # Validate position with risk manager
            validation = self.risk_manager.validate_new_position(
                symbol, signal_strength, self.positions, self.portfolio_value, stock_data
            )
            
            if not validation['approved']:
                return {
                    'success': False,
                    'error': 'Position rejected by risk manager',
                    'reasons': validation['rejection_reasons'],
                    'warnings': validation['warnings']
                }
            
            # Execute the trade
            current_price = stock_data['close'].iloc[-1]
            shares = validation['shares']
            position_value = validation['position_value']
            
            # Check if we have enough cash
            if position_value > self.cash_balance:
                return {'success': False, 'error': 'Insufficient cash balance'}
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=shares if signal_strength > 0 else -shares,
                entry_price=current_price,
                entry_time=datetime.now(),
                stop_loss=validation['stop_loss'],
                take_profit=validation['take_profit'],
                position_value=position_value,
                risk_amount=validation['risk_amount'],
                position_type='long' if signal_strength > 0 else 'short'
            )
            
            # Update portfolio
            self.positions[symbol] = position
            self.cash_balance -= position_value
            self.risk_manager.risk_budget_used += validation['risk_amount'] / self.portfolio_value
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'action': 'buy' if signal_strength > 0 else 'sell',
                'quantity': abs(shares),
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'signal_strength': signal_strength,
                'position_value': position_value,
                'risk_amount': validation['risk_amount'],
                'stop_loss': validation['stop_loss'],
                'take_profit': validation['take_profit']
            }
            
            self.trade_history.append(trade_record)
            self._save_trade_to_db(trade_record)
            self._save_position_to_db(position)
            
            self.performance_metrics['total_trades'] += 1
            
            self.logger.info(f"Executed trade: {trade_record['action'].upper()} {shares:.2f} shares of {symbol} at ${current_price:.2f}")
            
            return {
                'success': True,
                'trade': trade_record,
                'validation_warnings': validation.get('warnings', []),
                'position_id': symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def manage_positions(self) -> List[Dict]:
        """
        Manage existing positions - check stop losses, take profits, and trailing stops
        """
        management_actions = []
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                current_data = self.data_manager.get_stock_data(symbol, period="1d")
                if current_data is None:
                    continue
                
                current_price = current_data['close'].iloc[-1]
                
                # Calculate current P&L
                if position.position_type == 'long':
                    pnl = (current_price - position.entry_price) * position.quantity
                    pnl_percent = (current_price - position.entry_price) / position.entry_price
                else:  # short
                    pnl = (position.entry_price - current_price) * abs(position.quantity)
                    pnl_percent = (position.entry_price - current_price) / position.entry_price
                
                action_taken = None
                
                # Check stop loss
                if ((position.position_type == 'long' and current_price <= position.stop_loss) or
                    (position.position_type == 'short' and current_price >= position.stop_loss)):
                    
                    action_taken = self._close_position(symbol, current_price, 'stop_loss')
                    management_actions.append(action_taken)
                
                # Check take profit
                elif ((position.position_type == 'long' and current_price >= position.take_profit) or
                      (position.position_type == 'short' and current_price <= position.take_profit)):
                    
                    action_taken = self._close_position(symbol, current_price, 'take_profit')
                    management_actions.append(action_taken)
                
                # Check for trailing stop adjustment
                elif position.position_type == 'long' and pnl_percent > 0.10:  # 10% profit
                    # Implement trailing stop
                    new_stop_loss = current_price * 0.95  # 5% trailing stop
                    if new_stop_loss > position.stop_loss:
                        position.stop_loss = new_stop_loss
                        self.logger.info(f"Updated trailing stop for {symbol} to ${new_stop_loss:.2f}")
                
                # Check for time-based exit (holding too long)
                holding_period = datetime.now() - position.entry_time
                if holding_period.days > 30:  # 30 days max holding period
                    action_taken = self._close_position(symbol, current_price, 'time_limit')
                    management_actions.append(action_taken)
                
                # Update daily P&L tracking
                if action_taken:
                    self.risk_manager.update_daily_pnl(action_taken['pnl_percent'])
                
            except Exception as e:
                self.logger.error(f"Error managing position {symbol}: {str(e)}")
                continue
        
        return management_actions
    
    def _close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Dict:
        """Close a position and record the trade"""
        try:
            position = self.positions[symbol]
            
            # Calculate P&L
            if position.position_type == 'long':
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # short
                pnl = (position.entry_price - exit_price) * abs(position.quantity)
            
            pnl_percent = pnl / position.position_value
            
            # Update portfolio
            exit_value = abs(position.quantity) * exit_price
            self.cash_balance += exit_value
            self.risk_manager.risk_budget_used -= position.risk_amount / self.portfolio_value
            
            # Update performance metrics
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            self.performance_metrics['total_pnl'] += pnl
            
            # Record the exit
            exit_record = {
                'symbol': symbol,
                'action': f'close_{position.position_type}',
                'quantity': abs(position.quantity),
                'exit_price': exit_price,
                'entry_price': position.entry_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'exit_reason': exit_reason,
                'holding_period': (datetime.now() - position.entry_time).days,
                'timestamp': datetime.now().isoformat()
            }
            
            # Remove position
            del self.positions[symbol]
            
            # Save to database
            self._update_position_in_db(symbol, exit_price, exit_reason, pnl)
            
            self.logger.info(f"Closed {position.position_type} position in {symbol}: P&L ${pnl:.2f} ({pnl_percent:.2%})")
            
            return exit_record
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            # Calculate current portfolio value
            current_portfolio_value = self.cash_balance
            position_values = {}
            
            for symbol, position in self.positions.items():
                try:
                    current_data = self.data_manager.get_stock_data(symbol, period="1d")
                    if current_data is not None:
                        current_price = current_data['close'].iloc[-1]
                        position_value = abs(position.quantity) * current_price
                        current_portfolio_value += position_value
                        position_values[symbol] = {
                            'current_price': current_price,
                            'position_value': position_value,
                            'unrealized_pnl': position_value - position.position_value,
                            'unrealized_pnl_percent': (position_value - position.position_value) / position.position_value
                        }
                except:
                    continue
            
            # Calculate performance metrics
            total_return = (current_portfolio_value - self.portfolio_value) / self.portfolio_value
            
            win_rate = 0
            if self.performance_metrics['total_trades'] > 0:
                win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            
            # Portfolio allocation
            cash_allocation = self.cash_balance / current_portfolio_value if current_portfolio_value > 0 else 1.0
            
            summary = {
                'portfolio_value': current_portfolio_value,
                'cash_balance': self.cash_balance,
                'cash_allocation': cash_allocation,
                'total_return': total_return,
                'total_return_percent': total_return * 100,
                'num_positions': len(self.positions),
                'positions': position_values,
                'performance_metrics': self.performance_metrics,
                'win_rate': win_rate,
                'risk_budget_used': self.risk_manager.risk_budget_used,
                'daily_pnl': self.risk_manager.daily_pnl,
                'is_trading_enabled': self.is_trading_enabled,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {str(e)}")
            return {'error': str(e)}
    
    def run_trading_cycle(self) -> Dict:
        """
        Run a complete trading cycle: scan, execute, manage
        """
        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'opportunities_found': 0,
            'trades_executed': 0,
            'positions_managed': 0,
            'errors': []
        }
        
        try:
            if not self.is_trading_enabled:
                cycle_results['status'] = 'trading_disabled'
                return cycle_results
            
            # 1. Manage existing positions first
            management_actions = self.manage_positions()
            cycle_results['positions_managed'] = len(management_actions)
            
            # 2. Scan for new opportunities
            opportunities = self.scan_opportunities()
            cycle_results['opportunities_found'] = len(opportunities)
            
            # 3. Execute trades for best opportunities
            trades_executed = 0
            for opportunity in opportunities[:3]:  # Limit to top 3 opportunities
                try:
                    result = self.execute_trade(opportunity['symbol'], opportunity)
                    if result['success']:
                        trades_executed += 1
                    else:
                        cycle_results['errors'].append(f"{opportunity['symbol']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    cycle_results['errors'].append(f"{opportunity['symbol']}: {str(e)}")
            
            cycle_results['trades_executed'] = trades_executed
            
            # 4. Update portfolio history
            self._save_portfolio_snapshot()
            
            # 5. Risk management checks
            if self.risk_manager.daily_loss_limit_hit:
                self.is_trading_enabled = False
                cycle_results['status'] = 'daily_limit_hit'
                self.logger.warning("Trading disabled due to daily loss limit")
            else:
                cycle_results['status'] = 'completed'
            
            self.logger.info(f"Trading cycle completed: {trades_executed} trades executed, {len(management_actions)} positions managed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            cycle_results['errors'].append(str(e))
            cycle_results['status'] = 'error'
        
        return cycle_results
    
    def _save_trade_to_db(self, trade_record: Dict):
        """Save trade record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (symbol, action, quantity, price, timestamp, signal_strength, conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_record['symbol'],
                trade_record['action'],
                trade_record['quantity'],
                trade_record['price'],
                trade_record['timestamp'],
                trade_record['signal_strength'],
                json.dumps(trade_record)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving trade to database: {str(e)}")
    
    def _save_position_to_db(self, position: Position):
        """Save position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (symbol, quantity, entry_price, entry_time, stop_loss, take_profit, position_type, status, risk_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol,
                position.quantity,
                position.entry_price,
                position.entry_time.isoformat(),
                position.stop_loss,
                position.take_profit,
                position.position_type,
                'open',
                position.risk_amount
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving position to database: {str(e)}")
    
    def _update_position_in_db(self, symbol: str, exit_price: float, exit_reason: str, pnl: float):
        """Update position in database when closed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions 
                SET exit_price = ?, exit_time = ?, status = ?, pnl = ?
                WHERE symbol = ? AND status = 'open'
            ''', (
                exit_price,
                datetime.now().isoformat(),
                f'closed_{exit_reason}',
                pnl,
                symbol
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating position in database: {str(e)}")
    
    def _save_portfolio_snapshot(self):
        """Save portfolio snapshot to database"""
        try:
            summary = self.get_portfolio_summary()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_history (timestamp, portfolio_value, cash_balance, num_positions, daily_pnl)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                summary['portfolio_value'],
                summary['cash_balance'],
                summary['num_positions'],
                self.risk_manager.daily_pnl
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio snapshot: {str(e)}")
    
    def enable_trading(self):
        """Enable trading"""
        self.is_trading_enabled = True
        self.logger.info("Trading enabled")
    
    def disable_trading(self):
        """Disable trading"""
        self.is_trading_enabled = False
        self.logger.info("Trading disabled")
    
    def reset_daily_limits(self):
        """Reset daily limits (call at start of each trading day)"""
        self.risk_manager.reset_daily_limits()
        self.is_trading_enabled = True

# Example usage and configuration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_trader.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize AI Trader
    trader = AITrader()
    
    # Add some stocks to watchlist
    trader.add_to_watchlist(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'AMZN'])
    
    # Run a trading cycle
    print("Running trading cycle...")
    results = trader.run_trading_cycle()
    print(f"Cycle results: {results}")
    
    # Get portfolio summary
    print("\nPortfolio Summary:")
    summary = trader.get_portfolio_summary()
    for key, value in summary.items():
        if key != 'positions':
            print(f"{key}: {value}")
    
    print(f"\nPositions: {len(summary.get('positions', {}))}")
    for symbol, pos_info in summary.get('positions', {}).items():
        print(f"  {symbol}: ${pos_info['position_value']:.2f} (P&L: {pos_info['unrealized_pnl_percent']:.2%})")