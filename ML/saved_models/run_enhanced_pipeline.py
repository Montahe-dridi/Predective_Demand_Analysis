# ML/run_enhanced_pipeline.py

import sys
import os
import argparse
import traceback
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.saved_models.main_enhanced_pipeline import EnhancedMLPipeline
from ML.config.model_config import ModelConfig

def main():
    """Main entry point for enhanced ML pipeline"""
    
    parser = argparse.ArgumentParser(description="Enhanced TRALIS ML Pipeline")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Limit number of records for testing (default: all data)")
    parser.add_argument("--models", nargs='+', 
                       choices=['regression', 'classification', 'timeseries', 'segmentation', 'all'],
                       default=['all'], help="Which model types to run")
    parser.add_argument("--tune", action='store_true', 
                       help="Enable hyperparameter tuning (slower)")
    parser.add_argument("--dashboards", action='store_true', default=True,
                       help="Generate interactive dashboards")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = EnhancedMLPipeline(sample_limit=args.sample)
        
        if 'all' in args.models:
            results = pipeline.run_complete_pipeline()
        else:
            # Run specific components
            results = pipeline.run_selective_pipeline(args.models)
        
        print("\n🎉 Pipeline execution completed successfully!")
        print(f"📊 Check results in: {pipeline.config.OUTPUT_DIR}")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()