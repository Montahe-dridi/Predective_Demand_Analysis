# main.py

import argparse
import sys
import os
from datetime import datetime

# ETL Pipeline Imports
from Pipelines.Dim_tables.run_dim_products import run_pipeline as run_dim_product
from Pipelines.Dim_tables.run_dim_customer import run_pipeline as run_dim_customer
from Pipelines.Dim_tables.run_dim_supplier import run_pipeline as run_dim_supplier
from Pipelines.Dim_tables.run_dim_location import run_pipeline as run_dim_location
from Pipelines.Dim_tables.run_dim_equipment import run_pipeline as run_dim_equipment
from Pipelines.Dim_tables.run_dim_freighttype import run_pipeline as run_dim_freighttype
from Pipelines.Dim_tables.run_dim_paymentterm import run_pipeline as run_dim_paymentterm
from Pipelines.Dim_tables.generate_dim_date import run_pipeline as run_dim_date
from Pipelines.Fact_tables.run_fact_shipment import run_pipeline as run_fact_shipment
from Pipelines.Fact_tables.run_fact_invoices import run_pipeline as run_fact_invoices
from Pipelines.analytics_tables.run_shipment_performance import run_shipment_performance_pipeline as run_shipment_performance
from Pipelines.analytics_tables.run_freight_cost_analysis import run_freight_cost_pipeline as run_freight_cost

# ML Pipeline Imports
try:
    from ML.saved_models.main_enhanced_pipeline import EnhancedMLPipeline
    from ML.pipelines.prediction_pipeline import PredictionPipeline
    from ML.config.model_config import ModelConfig
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML modules not available: {e}")
    ML_AVAILABLE = False

def run_ml_complete_pipeline(sample_limit=None):
    """Run the complete ML pipeline"""
    if not ML_AVAILABLE:
        print("❌ ML pipeline not available. Check ML module installation.")
        return
    
    try:
        print("🚀 Starting TRALIS ML Analytics Pipeline...")
        pipeline = EnhancedMLPipeline(sample_limit=sample_limit)
        results = pipeline.run_complete_pipeline()
        
        print("🎉 ML Pipeline completed successfully!")
        print(f"📊 Results saved to: {pipeline.config.OUTPUT_DIR}")
        return results
        
    except Exception as e:
        print(f"❌ ML Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_ml_selective_pipeline(models_list, sample_limit=None):
    """Run selective ML pipeline components"""
    if not ML_AVAILABLE:
        print("❌ ML pipeline not available. Check ML module installation.")
        return
    
    try:
        print(f"🎯 Starting selective ML pipeline: {models_list}")
        pipeline = EnhancedMLPipeline(sample_limit=sample_limit)
        results = pipeline.run_selective_pipeline(models_list)
        
        print(f"✅ Selective ML Pipeline completed for: {models_list}")
        return results
        
    except Exception as e:
        print(f"❌ Selective ML Pipeline failed: {e}")
        return None

def run_ml_predictions(data_limit=None):
    """Run ML predictions on existing data"""
    if not ML_AVAILABLE:
        print("❌ ML prediction pipeline not available.")
        return
    
    try:
        print("🔮 Starting ML Prediction Pipeline...")
        
        predictor = PredictionPipeline()
        
        # Load production models
        load_results = predictor.load_production_models()
        loaded_count = sum(load_results.values())
        
        if loaded_count == 0:
            print("❌ No production models found. Run training pipeline first.")
            return
        
        print(f"✅ Loaded {loaded_count} production models")
        
        # Load recent data for predictions
        from ML.data.loaders import DataLoader
        loader = DataLoader()
        
        shipments = loader.load_shipments(limit=data_limit)
        invoices = loader.load_invoices(limit=data_limit)
        
        if not shipments.empty:
            # Predict delivery performance
            delivery_predictions = predictor.predict_delivery_performance(shipments)
            
            # Assess shipment risks
            risk_assessment = predictor.predict_shipment_risks(shipments)
            
            # Save predictions
            config = ModelConfig()
            delivery_predictions.to_csv(os.path.join(config.OUTPUT_DIR, "delivery_predictions.csv"), index=False)
            risk_assessment.to_csv(os.path.join(config.OUTPUT_DIR, "shipment_risk_assessment.csv"), index=False)
            
            print(f"✅ Delivery predictions saved for {len(delivery_predictions)} shipments")
        
        if not invoices.empty:
            # Predict invoice payment
            payment_predictions = predictor.predict_invoice_payment(invoices)
            
            # Save predictions
            payment_predictions.to_csv(os.path.join(config.OUTPUT_DIR, "payment_predictions.csv"), index=False)
            
            print(f"✅ Payment predictions saved for {len(payment_predictions)} invoices")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Prediction Pipeline failed: {e}")
        return None

def run_etl_then_ml():
    """Run complete ETL pipeline followed by ML pipeline"""
    
    print("🔄 Running Complete ETL + ML Pipeline")
    print("=" * 50)
    
    # Step 1: Run all dimension tables
    print("\n📊 PHASE 1: Loading Dimension Tables")
    dimension_tables = [
        ("dim_date", run_dim_date),
        ("dim_customer", run_dim_customer),
        ("dim_supplier", run_dim_supplier),
        ("dim_product", run_dim_product),
        ("dim_location", run_dim_location),
        ("dim_equipment", run_dim_equipment),
        ("dim_freighttype", run_dim_freighttype),
        ("dim_paymentterm", run_dim_paymentterm)
    ]
    
    for table_name, pipeline_func in dimension_tables:
        try:
            print(f"  🔄 Running {table_name}...")
            pipeline_func()
            print(f"  ✅ {table_name} completed")
        except Exception as e:
            print(f"  ❌ {table_name} failed: {e}")
            continue
    
    # Step 2: Run fact tables
    print("\n📦 PHASE 2: Loading Fact Tables")
    fact_tables = [
        ("fact_shipments", run_fact_shipment),
        ("fact_invoices", run_fact_invoices)
    ]
    
    for table_name, pipeline_func in fact_tables:
        try:
            print(f"  🔄 Running {table_name}...")
            pipeline_func()
            print(f"  ✅ {table_name} completed")
        except Exception as e:
            print(f"  ❌ {table_name} failed: {e}")
            continue
    
    # Step 3: Run analytics tables
    print("\n📈 PHASE 3: Analytics Tables")
    analytics_tables = [
        ("shipment_performance", run_shipment_performance),
        ("freight_cost", run_freight_cost)
    ]
    
    for table_name, pipeline_func in analytics_tables:
        try:
            print(f"  🔄 Running {table_name}...")
            pipeline_func()
            print(f"  ✅ {table_name} completed")
        except Exception as e:
            print(f"  ❌ {table_name} failed: {e}")
            continue
    
    # Step 4: Run ML Pipeline
    print("\n🤖 PHASE 4: Machine Learning Analytics")
    if ML_AVAILABLE:
        try:
            ml_results = run_ml_complete_pipeline(sample_limit=None)
            if ml_results:
                print("  ✅ ML Pipeline completed successfully")
            else:
                print("  ⚠️ ML Pipeline completed with issues")
        except Exception as e:
            print(f"  ❌ ML Pipeline failed: {e}")
    else:
        print("  ⚠️ ML Pipeline not available")
    
    print("\n🎉 Complete ETL + ML Pipeline Finished!")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_help():
    """Print comprehensive help information"""
    
    help_text = """
🚀 TRALIS ETL + ML Pipeline Runner

📊 ETL COMMANDS (Data Warehouse):
  --table dim_product           Load product dimension
  --table dim_customer          Load customer dimension  
  --table dim_supplier          Load supplier dimension
  --table dim_location          Load location dimension
  --table dim_equipment         Load equipment dimension
  --table dim_freighttype       Load freight type dimension
  --table dim_paymentterm       Load payment term dimension
  --table dim_date              Generate date dimension
  --table fact_shipments        Load shipment facts
  --table fact_invoices         Load invoice facts
  --table shipment_performance  Generate shipment analytics
  --table freight_cost          Generate freight cost analytics

🤖 ML COMMANDS (Machine Learning):
  --table ml_complete           Run complete ML pipeline (all models)
  --table ml_regression         Run only regression models
  --table ml_classification     Run only classification models  
  --table ml_timeseries         Run only time series forecasting
  --table ml_segmentation       Run only customer segmentation
  --table ml_predictions        Run predictions on existing data
  --table ml_quick              Quick ML test (sample data, no tuning)

🔄 COMBINED COMMANDS:
  --table etl_then_ml           Run complete ETL followed by ML
  --table all_dimensions        Run all dimension table loads
  --table all_facts             Run all fact table loads

⚙️ ML OPTIONS:
  --sample N                    Limit ML to N records (for testing)
  --tune                        Enable hyperparameter tuning (slower)
  --models MODEL1 MODEL2        Specify which ML models to run

📋 EXAMPLES:
  python main.py --table dim_customer                    # Load customer dimension
  python main.py --table ml_complete                     # Run complete ML pipeline
  python main.py --table ml_quick --sample 1000          # Quick ML test
  python main.py --table ml_regression --tune            # Regression with tuning
  python main.py --table etl_then_ml                     # Complete ETL + ML
  python main.py --table ml_predictions                  # Generate predictions
"""
    
    print(help_text)

def main():
    parser = argparse.ArgumentParser(
        description="ETL + ML Runner for TRALIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help manually
    )
    
    parser.add_argument("--table", required=False, help="Name of the pipeline to run")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for ML pipeline")
    parser.add_argument("--tune", action='store_true', help="Enable ML hyperparameter tuning")
    parser.add_argument("--models", nargs='+', 
                       choices=['regression', 'classification', 'timeseries', 'segmentation'],
                       help="Specific ML models to run")
    parser.add_argument("--help", "-h", action='store_true', help="Show help message")
    
    args = parser.parse_args()
    
    # Handle help
    if args.help or not args.table:
        print_help()
        return
    
    print(f"🚀 TRALIS Pipeline Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Running: {args.table}")
    
    # ETL Pipelines
    if args.table == "dim_product":
        run_dim_product()
    elif args.table == "dim_customer":
        run_dim_customer()
    elif args.table == "dim_supplier":
        run_dim_supplier()
    elif args.table == "dim_location":
        run_dim_location()
    elif args.table == "dim_equipment":
        run_dim_equipment()
    elif args.table == "dim_freighttype":
        run_dim_freighttype()
    elif args.table == "dim_paymentterm":
        run_dim_paymentterm()   
    elif args.table == "dim_date":
        run_dim_date()
    elif args.table == "fact_shipments":
        run_fact_shipment()
    elif args.table == "fact_invoices":
        run_fact_invoices()
    elif args.table == "shipment_performance":
        run_shipment_performance()
    elif args.table == "freight_cost":
        run_freight_cost()
    
    # Combined ETL Commands
    elif args.table == "all_dimensions":
        print("📊 Running all dimension table pipelines...")
        dimension_pipelines = [
            ("dim_date", run_dim_date),
            ("dim_customer", run_dim_customer),
            ("dim_supplier", run_dim_supplier),
            ("dim_product", run_dim_product),
            ("dim_location", run_dim_location),
            ("dim_equipment", run_dim_equipment),
            ("dim_freighttype", run_dim_freighttype),
            ("dim_paymentterm", run_dim_paymentterm)
        ]
        
        for name, pipeline in dimension_pipelines:
            try:
                print(f"  🔄 Running {name}...")
                pipeline()
                print(f"  ✅ {name} completed")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
    
    elif args.table == "all_facts":
        print("📦 Running all fact table pipelines...")
        fact_pipelines = [
            ("fact_shipments", run_fact_shipment),
            ("fact_invoices", run_fact_invoices),
            ("shipment_performance", run_shipment_performance),
            ("freight_cost", run_freight_cost)
        ]
        
        for name, pipeline in fact_pipelines:
            try:
                print(f"  🔄 Running {name}...")
                pipeline()
                print(f"  ✅ {name} completed")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
    
    # ML Pipelines
    elif args.table == "ml_complete":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available. Install ML dependencies.")
            return
        
        print("🤖 Running Complete ML Pipeline...")
        if args.tune:
            print("⚙️ Hyperparameter tuning enabled (this will take longer)")
        if args.sample:
            print(f"📊 Using sample size: {args.sample:,} records")
        
        run_ml_complete_pipeline(sample_limit=args.sample)
    
    elif args.table == "ml_regression":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available.")
            return
        
        print("📈 Running ML Regression Models...")
        run_ml_selective_pipeline(['regression'], sample_limit=args.sample)
    
    elif args.table == "ml_classification":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available.")
            return
        
        print("🎯 Running ML Classification Models...")
        run_ml_selective_pipeline(['classification'], sample_limit=args.sample)
    
    elif args.table == "ml_timeseries":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available.")
            return
        
        print("📊 Running Time Series Forecasting...")
        run_ml_selective_pipeline(['timeseries'], sample_limit=args.sample)
    
    elif args.table == "ml_segmentation":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available.")
            return
        
        print("👥 Running Customer Segmentation...")
        run_ml_selective_pipeline(['segmentation'], sample_limit=args.sample)
    
    elif args.table == "ml_predictions":
        if not ML_AVAILABLE:
            print("❌ ML prediction pipeline not available.")
            return
        
        print("🔮 Running ML Predictions...")
        run_ml_predictions(data_limit=args.sample)
    
    elif args.table == "ml_quick":
        if not ML_AVAILABLE:
            print("❌ ML pipeline not available.")
            return
        
        print("⚡ Running Quick ML Test...")
        sample_size = args.sample or 1000
        run_ml_selective_pipeline(['regression', 'classification'], sample_limit=sample_size)
    
    # Combined ETL + ML Pipeline
    elif args.table == "etl_then_ml":
        print("🔄 Running Complete ETL + ML Pipeline...")
        run_etl_then_ml()
    
    # Unknown command
    else:
        print(f"❌ Unknown pipeline: {args.table}")
        print("\n📋 Available ETL pipelines:")
        etl_options = [
            "dim_product", "dim_customer", "dim_supplier", "dim_location",
            "dim_equipment", "dim_freighttype", "dim_paymentterm", "dim_date",
            "fact_shipments", "fact_invoices", "shipment_performance", "freight_cost"
        ]
        print(f"   {', '.join(etl_options)}")
        
        if ML_AVAILABLE:
            print("\n🤖 Available ML pipelines:")
            ml_options = [
                "ml_complete", "ml_regression", "ml_classification", "ml_timeseries",
                "ml_segmentation", "ml_predictions", "ml_quick"
            ]
            print(f"   {', '.join(ml_options)}")
        
        print("\n🔄 Combined pipelines:")
        combined_options = ["all_dimensions", "all_facts", "etl_then_ml"]
        print(f"   {', '.join(combined_options)}")
        
        print(f"\n💡 Use --help for detailed usage examples")

def check_ml_dependencies():
    """Check if ML dependencies are available"""
    
    required_ml_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'plotly']
    missing_packages = []
    
    for package in required_ml_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing ML packages: {missing_packages}")
        print(f"📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # Check for help flag first
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_help()
        sys.exit(0)
    
    # Check ML dependencies if ML command is used
    if len(sys.argv) > 1 and any('ml_' in arg for arg in sys.argv):
        if not check_ml_dependencies():
            print("❌ Cannot run ML pipeline without required dependencies")
            sys.exit(1)
    
    main()