import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf

def calculate_returns(price_df: pd.DataFrame, rolling_window: int = 21) -> dict:
    """Calculates daily returns, log returns, and rolling cumulative returns."""
    returns_dict = {}
    
    # 1. Daily Returns
    returns_dict['daily_returns'] = price_df.pct_change().dropna()
    
    # 2. Log Returns
    returns_dict['log_returns'] = np.log(price_df / price_df.shift(1)).dropna()
    
    # 3. Rolling Cumulative Returns (Sum of daily returns over the window)
    returns_dict['rolling_returns'] = returns_dict['daily_returns'].rolling(window=rolling_window).sum().dropna()
    
    return returns_dict
def estimate_covariance(returns_df: pd.DataFrame) -> dict:
    """Computes Sample Covariance, Correlation, and Ledoit-Wolf Shrinkage matrices."""
    cov_dict = {}
    
    # 1. Sample Covariance Matrix
    cov_dict['sample_covariance'] = returns_df.cov()
    
    # 2. Correlation Matrix
    cov_dict['correlation_matrix'] = returns_df.corr()
    
    # 3. Ledoit-Wolf Shrinkage (Reduces estimation errors in small samples)
    lw = LedoitWolf()
    lw.fit(returns_df)
    cov_dict['ledoit_wolf_covariance'] = pd.DataFrame(
        lw.covariance_, 
        index=returns_df.columns, 
        columns=returns_df.columns
    )
    
    return cov_dict
def generate_risk_visualizations(returns_df: pd.DataFrame, cov_dict: dict):
    """Renders required heatmaps for portfolio risk profiling."""
    # 1. Annualized Volatility Heatmap
    annualized_vol = returns_df.std() * np.sqrt(252) # 252 trading days/year
    vol_df = pd.DataFrame(annualized_vol, columns=['Annualized Volatility'])
    
    plt.figure(figsize=(4, 5))
    sns.heatmap(vol_df, annot=True, cmap='YlOrRd', fmt=".2%")
    plt.title('Asset Volatility Profile')
    plt.tight_layout()
    plt.savefig('notebooks/volatility_heatmap.png') # Saves directly to deliverables folder
    plt.show()

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(cov_dict['correlation_matrix'], annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('notebooks/correlation_heatmap.png')
    plt.show()

    # 3. Covariance Visualization (Ledoit-Wolf Matrix)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cov_dict['ledoit_wolf_covariance'], annot=True, cmap='Blues', fmt=".6f")
    plt.title('Ledoit-Wolf Shrunk Covariance Matrix')
    plt.tight_layout()
    plt.savefig('notebooks/covariance_heatmap.png')
    plt.show()
if __name__ == "__main__":
    # Generate 100 days of mock business stock data for 3 assets
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq='B')
    mock_prices = pd.DataFrame({
        'Tech_Stock': 100 * np.exp(np.random.normal(0.0005, 0.01, 100).cumsum()),
        'Bond_Index': 50 * np.exp(np.random.normal(0.0001, 0.002, 100).cumsum()),
        'Gold_ETF': 150 * np.exp(np.random.normal(0.0002, 0.005, 100).cumsum())
    }, index=dates)

    # Execute the analytics framework step-by-step
    print("1. Running asset return engines...")
    returns = calculate_returns(mock_prices)
    
    print("2. Mapping portfolio risk footprints...")
    covariances = estimate_covariance(returns['daily_returns'])
    
    print("3. Exporting processed analytics tables to data/ dir...")
    returns['daily_returns'].to_csv('data/processed_daily_returns.csv')
    covariances['ledoit_wolf_covariance'].to_csv('data/covariance_matrix.csv')
    
    print("4. Rendering and logging graphic deliverables...")
    generate_risk_visualizations(returns['daily_returns'], covariances)
    print("\n[SUCCESS]: Framework task complete. Check data/ and notebooks/ folders.")
