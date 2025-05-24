import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, List
import logging

class PricePredictionModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.scaler = MinMaxScaler()
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='huber',  # More robust to outliers
            metrics=['mae']
        )
        
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model."""
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Add early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Add learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            )
            
            history = self.model.fit(
                X_scaled, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            return history
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return None
            
    def predict(self, X):
        """Make predictions."""
        try:
            if X is None or len(X) == 0:
                logging.error("Empty input data for prediction")
                return None
                
            # Ensure X is a numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
                
            # Scale the input data
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Ensure predictions are valid
            if predictions is None or len(predictions) == 0:
                logging.error("No predictions generated")
                return None
                
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Select features
        features = ['close', 'volume', 'rsi', 'macd', 'bbands_upper', 'bbands_lower']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.input_shape[0]):
            X.append(scaled_data[i:(i + self.input_shape[0])])
            y.append(scaled_data[i + self.input_shape[0], 0])  # Predict next close price
        
        return np.array(X), np.array(y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        try:
            metrics = self.model.evaluate(X, y, verbose=0)
            return {
                'loss': metrics[0],
                'mae': metrics[1]
            }
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            return {'loss': float('inf'), 'mae': float('inf')}
    
    def save_model(self, filepath: str):
        """Save model weights."""
        try:
            self.model.save_weights(filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        try:
            self.model.load_weights(filepath)
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    
    def generate_trading_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals based on model predictions."""
        try:
            # Prepare data
            X, _ = self.prepare_data(df)
            
            # Get predictions
            predictions = []
            for i in range(len(X)):
                pred = self.predict(X[i])
                predictions.append(pred[0, 0])
            
            # Calculate returns
            actual_returns = df['close'].pct_change()
            predicted_returns = pd.Series(predictions, index=df.index[self.input_shape[0]:]).pct_change()
            
            # Generate signals
            entry_signals = pd.Series(False, index=df.index)
            exit_signals = pd.Series(False, index=df.index)
            
            # Entry signals
            entry_signals[predicted_returns > 0.001] = True  # 0.1% threshold
            
            # Exit signals
            exit_signals[predicted_returns < -0.001] = True  # -0.1% threshold
            
            return entry_signals, exit_signals
            
        except Exception as e:
            logging.error(f"Error generating trading signals: {e}")
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index) 