# ML/config/feature_config.py

class FeatureConfig:
    """Feature engineering configuration"""
    
    SHIPMENT_FEATURES = {
        'base': ['TotalWeight', 'TotalVolume', 'TotalPackages', 'ShipmentValue'],
        'temporal': ['Month', 'DayOfWeek', 'Quarter', 'Year'],
        'categorical': ['CustomerKey', 'OriginLocationKey', 'DestinationLocationKey', 
                       'EquipmentKey', 'FreightTypeKey'],
        'engineered': ['WeightVolumeDensity', 'ValuePerWeight', 'ValuePerPackage']
    }
    
    INVOICE_FEATURES = {
        'base': ['TotalAmount', 'NetAmount', 'TaxAmount'],
        'temporal': ['Month', 'Quarter', 'DayOfWeek', 'Year'],
        'categorical': ['CustomerKey', 'SupplierKey', 'PaymentTermKey'],
        'engineered': ['TaxRate', 'AmountPerDay', 'ProfitMargin']
    }
    
    SCALING_METHODS = {
        'robust': 'RobustScaler',
        'standard': 'StandardScaler', 
        'minmax': 'MinMaxScaler'
    }