import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedAnalysis:
    @staticmethod
    def test_stationarity(series: pd.Series) -> Dict[str, float]:
        """Test time series stationarity using Augmented Dickey-Fuller test."""
        try:
            result = adfuller(series.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            logging.error(f"Error testing stationarity: {e}")
            return {}
    
    @staticmethod
    def decompose_time_series(series: pd.Series, period: int = 24) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components."""
        try:
            # Calculate moving average
            trend = series.rolling(window=period).mean()
            
            # Calculate seasonal component
            detrended = series - trend
            seasonal = detrended.groupby(detrended.index.hour).mean()
            seasonal = pd.Series(seasonal.values, index=series.index)
            
            # Calculate residual
            residual = series - trend - seasonal
            
            return {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }
        except Exception as e:
            logging.error(f"Error decomposing time series: {e}")
            return {}
    
    @staticmethod
    def fit_arima_model(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """Fit ARIMA model to time series."""
        try:
            model = ARIMA(series, order=order)
            results = model.fit()
            
            return {
                'model': results,
                'aic': results.aic,
                'bic': results.bic,
                'forecast': results.forecast(steps=24)
            }
        except Exception as e:
            logging.error(f"Error fitting ARIMA model: {e}")
            return {}
    
    @staticmethod
    def perform_pca_analysis(data: pd.DataFrame, n_components: int = 3) -> Dict:
        """Perform Principal Component Analysis."""
        try:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            
            return {
                'pca_result': pca_result,
                'explained_variance': explained_variance,
                'components': pca.components_,
                'feature_importance': pd.DataFrame(
                    pca.components_,
                    columns=data.columns
                )
            }
        except Exception as e:
            logging.error(f"Error performing PCA: {e}")
            return {}
    
    @staticmethod
    def perform_clustering(data: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """Perform K-means clustering."""
        try:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Calculate cluster centers
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            return {
                'clusters': clusters,
                'centers': centers,
                'inertia': kmeans.inertia_,
                'silhouette_score': silhouette_score(scaled_data, clusters)
            }
        except Exception as e:
            logging.error(f"Error performing clustering: {e}")
            return {}
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with p-values."""
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Calculate p-values
            p_values = pd.DataFrame(
                np.zeros_like(corr_matrix),
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )
            
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if i != j:
                        corr, p_value = stats.pearsonr(
                            data[corr_matrix.columns[i]],
                            data[corr_matrix.columns[j]]
                        )
                        p_values.iloc[i, j] = p_value
            
            return {
                'correlation': corr_matrix,
                'p_values': p_values
            }
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    @staticmethod
    def perform_volatility_analysis(returns: pd.Series) -> Dict[str, float]:
        """Perform volatility analysis."""
        try:
            # Calculate volatility metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            downside_volatility = returns[returns < 0].std() * np.sqrt(252)
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate Expected Shortfall (ES)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            return {
                'volatility': volatility,
                'downside_volatility': downside_volatility,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99
            }
        except Exception as e:
            logging.error(f"Error performing volatility analysis: {e}")
            return {}
    
    @staticmethod
    def analyze_market_regime(returns: pd.Series, window: int = 20) -> Dict[str, str]:
        """Analyze market regime using rolling statistics."""
        try:
            # Calculate rolling metrics
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            
            # Calculate rolling Sharpe ratio
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
            
            # Determine market regime
            current_sharpe = rolling_sharpe.iloc[-1]
            current_vol = rolling_std.iloc[-1]
            
            if current_sharpe > 1:
                regime = 'BULLISH'
            elif current_sharpe < -1:
                regime = 'BEARISH'
            else:
                regime = 'NEUTRAL'
            
            volatility_state = 'HIGH' if current_vol > returns.std() else 'LOW'
            
            return {
                'regime': regime,
                'volatility_state': volatility_state,
                'sharpe_ratio': current_sharpe,
                'volatility': current_vol
            }
        except Exception as e:
            logging.error(f"Error analyzing market regime: {e}")
            return {}
    
    @staticmethod
    def plot_analysis_results(data: Dict, title: str = "Analysis Results"):
        """Plot analysis results."""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot time series decomposition
            if 'trend' in data:
                plt.subplot(3, 1, 1)
                plt.plot(data['trend'], label='Trend')
                plt.title('Trend Component')
                plt.legend()
                
                plt.subplot(3, 1, 2)
                plt.plot(data['seasonal'], label='Seasonal')
                plt.title('Seasonal Component')
                plt.legend()
                
                plt.subplot(3, 1, 3)
                plt.plot(data['residual'], label='Residual')
                plt.title('Residual Component')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting analysis results: {e}")

if __name__ == "__main__":
    print("تم تشغيل ملف advanced_analysis.py بنجاح") 