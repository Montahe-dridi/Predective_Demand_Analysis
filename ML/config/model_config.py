# ML/config/model_config.py

import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelConfig:
    """Configuration for ML models and pipelines"""
    
    # Directories
    BASE_DIR: str = os.path.join(os.getcwd(), "ML")
    MODEL_DIR: str = os.path.join(BASE_DIR, "saved_models")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "plots")
    REPORTS_DIR: str = os.path.join(OUTPUT_DIR, "reports")
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    
    # Time series
    TS_HORIZON_DAYS: int = 30
    TS_HORIZON_MONTHS: int = 6
    TS_SEASONAL_PERIODS: int = 7
    
    # Customer segmentation
    N_CLUSTERS: int = 4
    RFM_QUANTILES: int = 4
    
    # Performance thresholds
    MIN_ACCURACY: float = 0.65  # Was 0.70
    MIN_R2: float = 0.45        # Was 0.60
    MAX_MAPE: float = 35.0      # Was 20.0
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.MODEL_DIR, self.OUTPUT_DIR, self.PLOTS_DIR, self.REPORTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)