from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

class OrderManager:
    def __init__(self, client: Client, config: Dict):
        self.client = client
        self.config = config
        self.pending_orders = {}
        self.active_orders = {}
        self.stop_loss_orders = {}
        self.take_profit_orders = {}

    def validate_signal(self, signal: Dict, market_data: pd.DataFrame) -> bool:
        """التحقق من صحة إشارة التداول"""
        if not signal.get('buy') and not signal.get('sell'):
            return False

        # التحقق من حجم التداول
        volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        if volume < avg_volume * 0.5:
            return False

        # التحقق من التقلبات
        volatility = market_data['close'].pct_change().std()
        if volatility > self.config.get('MAX_VOLATILITY', 0.05):
            return False

        # التحقق من مستوى الثقة
        if signal.get('confidence', 0) < self.config.get('MIN_SIGNAL_CONFIDENCE', 0.6):
            return False

        return True

    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = 'MARKET', price: Optional[float] = None) -> Dict:
        """وضع أمر تداول"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price if order_type == 'LIMIT' else None
            )
            
            if order_type == 'MARKET':
                self.active_orders[order['orderId']] = order
            else:
                self.pending_orders[order['orderId']] = order
                
            return order
            
        except BinanceAPIException as e:
            print(f"Error placing order: {e}")
            return None

    def place_stop_loss(self, symbol: str, quantity: float, 
                       stop_price: float, order_id: str) -> Dict:
        """وضع أمر وقف الخسارة"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='STOP_LOSS_LIMIT',
                quantity=quantity,
                stopPrice=stop_price,
                price=stop_price * 0.99  # سعر التنفيذ أقل قليلاً من سعر الوقف
            )
            
            self.stop_loss_orders[order_id] = order
            return order
            
        except BinanceAPIException as e:
            print(f"Error placing stop loss: {e}")
            return None

    def place_take_profit(self, symbol: str, quantity: float, 
                         take_profit_price: float, order_id: str) -> Dict:
        """وضع أمر جني الأرباح"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                quantity=quantity,
                price=take_profit_price
            )
            
            self.take_profit_orders[order_id] = order
            return order
            
        except BinanceAPIException as e:
            print(f"Error placing take profit: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """إلغاء أمر معلق"""
        try:
            self.client.cancel_order(orderId=order_id)
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            return True
        except BinanceAPIException as e:
            print(f"Error canceling order: {e}")
            return False

    def update_order_status(self, order_id: str) -> Dict:
        """تحديث حالة الأمر"""
        try:
            order = self.client.get_order(orderId=order_id)
            
            if order['status'] == 'FILLED':
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                    
            return order
            
        except BinanceAPIException as e:
            print(f"Error updating order status: {e}")
            return None

    def manage_trailing_stop(self, symbol: str, order_id: str, 
                           current_price: float) -> None:
        """إدارة وقف الخسارة المتحرك"""
        if order_id in self.stop_loss_orders:
            stop_order = self.stop_loss_orders[order_id]
            stop_price = float(stop_order['stopPrice'])
            
            # تحديث سعر الوقف إذا ارتفع السعر
            if current_price > stop_price * 1.02:  # 2% أعلى من سعر الوقف الحالي
                new_stop_price = current_price * 0.99  # 1% أقل من السعر الحالي
                
                # إلغاء الأمر القديم
                self.cancel_order(stop_order['orderId'])
                
                # وضع أمر جديد
                self.place_stop_loss(
                    symbol=symbol,
                    quantity=float(stop_order['origQty']),
                    stop_price=new_stop_price,
                    order_id=order_id
                )