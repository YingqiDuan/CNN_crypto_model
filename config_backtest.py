"""
Configuration file for the backtest system.
All parameters are centralized here for easy adjustment.
"""

# Backtesting parameters
BACKTEST_CONFIG = {
    "initial_capital": 50.0,  # Initial capital for backtesting
    "position_size": 1,  # Portion of capital to use per trade (1 = 100%)
    "fee_rate": 0.0005,  # Trading fee rate (0.05%)
    "max_hold_periods": 1,  # Maximum number of periods to hold a position
}

# Strategy parameters
STRATEGY_CONFIG = {
    "ema_window": 6,  # EMA window size for f5 strategy
    "sma_window": 6,  # SMA window size for f5 strategy
}

# Data directory configuration
DATA_CONFIG = {
    "data_dir": "merged_csv",  # Directory containing CSV files
    "file_pattern": "*_4h_*.csv",  # Pattern to match CSV files
}

# Results directory configuration
RESULTS_CONFIG = {
    "results_dir": "backtest/backtest_result_f5",  # Main results directory
    "logs_dir": "backtest/backtest_result_f5/logs",  # Directory for log files
}

# Multiprocessing configuration
# If physical_cpu_multiplier is set to None, it uses 75% of logical cores
MULTIPROCESSING_CONFIG = {
    "physical_cpu_multiplier": 1.0,  # Multiplier for physical CPU count (None to use logical cores)
    "logical_cpu_percent": 0.75,  # Percentage of logical cores to use if physical count isn't available
}
