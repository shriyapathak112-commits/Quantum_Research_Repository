import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import DataIngestionPipeline

# 1. Setup paths
BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
output_path = os.path.join(PROCESSED_DIR, "processed_log_returns.csv")

print("🔄 Loading pipeline and aggregating data...")
# 2. Extract and combine categories using your pipeline class
pipeline = DataIngestionPipeline(base_dir=BASE_DIR)
etfs_df = pipeline.aggregate_category("etfs")
stocks_df = pipeline.aggregate_category("stocks")
macro_df = pipeline.aggregate_category("macro")

price_matrices = [df for df in [etfs_df, stocks_df, macro_df] if not df.empty]

if not price_matrices:
    print("❌ Error: No data found in your data/ folder. Please run your downloader first!")
else:
    master_price_df = pd.concat(price_matrices, axis=1)

    print("🔢 Calculating log returns...")
    # 3. Apply mathematical transformation
    log_returns_df = np.log(master_price_df) - np.log(master_price_df.shift(1))
    log_returns_df = log_returns_df.dropna(how="all").ffill().bfill()

    # 4. Save dataset
    log_returns_df.to_csv(output_path)
    print(f"🎉 Processed dataset saved to {output_path}")

    # 5. Generate and save the visualization chart
    print("📊 Generating Log Return Distributions plot...")
    plt.figure(figsize=(10, 5))
    
    # Target tickers available in your dataframe
    tickers = log_returns_df.columns[:3] # Plot first 3 assets as a sample
    for ticker in tickers:
        sns.kdeplot(log_returns_df[ticker], label=ticker, fill=True, alpha=0.2)
        
    plt.title("Daily Log Return Distributions")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    
    # Save chart to disk since Jupyter window can't show it
    plot_path = os.path.join(PROCESSED_DIR, "return_distributions.png")
    plt.savefig(plot_path)
    print(f"📉 Plot successfully saved as an image to {plot_path}")
    plt.close()
