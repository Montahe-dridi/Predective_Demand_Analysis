# ================================
# ML/enhanced_customer_segmentation.py
# ================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Tuple, List
from ML.config.model_config import ModelConfig

class AdvancedCustomerSegmentation:
    """Enhanced customer segmentation with multiple algorithms"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.cluster_models = {}
    
    def compute_enhanced_rfm(self, invoices_df: pd.DataFrame, 
                           snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
        """Enhanced RFM calculation with additional business metrics"""
        
        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now()
        
        df = invoices_df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Basic RFM
        rfm = df.groupby('CustomerKey').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceID': 'nunique',
            'NetAmount': ['sum', 'mean', 'std']
        })
        
        # Flatten columns
        rfm.columns = ['Recency', 'Frequency', 'Monetary_sum', 'Monetary_mean', 'Monetary_std']
        rfm['Monetary'] = rfm['Monetary_sum']
        rfm['Monetary_std'] = rfm['Monetary_std'].fillna(0)
        
        # Additional metrics
        customer_details = df.groupby('CustomerKey').agg({
            'InvoiceDate': ['min', 'max'],
            'PaymentStatus': lambda x: (x == 1).mean() if len(x) > 0 else 0,
            'TotalAmount': 'sum'
        })
        
        customer_details.columns = ['FirstPurchase', 'LastPurchase', 'PaymentRate', 'TotalSpent']
        
        # Customer lifetime metrics
        customer_details['CustomerLifetime'] = (customer_details['LastPurchase'] - 
                                              customer_details['FirstPurchase']).dt.days
        customer_details['AvgOrderValue'] = customer_details['TotalSpent'] / rfm['Frequency']
        customer_details['PurchaseVelocity'] = rfm['Frequency'] / (customer_details['CustomerLifetime'] + 1)
        
        # Combine all metrics
        enhanced_rfm = rfm.join(customer_details)
        enhanced_rfm = enhanced_rfm.fillna(0)
        
        # RFM scoring
        enhanced_rfm['R_score'] = pd.qcut(enhanced_rfm['Recency'].rank(method='first'), 
                                         q=5, labels=[5,4,3,2,1]).astype(int)
        enhanced_rfm['F_score'] = pd.qcut(enhanced_rfm['Frequency'].rank(method='first'), 
                                         q=5, labels=[1,2,3,4,5]).astype(int)
        enhanced_rfm['M_score'] = pd.qcut(enhanced_rfm['Monetary'].rank(method='first'), 
                                         q=5, labels=[1,2,3,4,5]).astype(int)
        
        # Customer value score
        enhanced_rfm['CustomerValue'] = (enhanced_rfm['F_score'] * enhanced_rfm['M_score'] * 
                                       enhanced_rfm['PaymentRate'])
        
        return enhanced_rfm.reset_index()
    
    def optimal_clustering(self, data: pd.DataFrame, 
                         features: List[str], max_clusters: int = 10) -> Dict:
        """Find optimal number of clusters using multiple metrics"""
        
        X = data[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        cluster_metrics = {}
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            inertia = kmeans.inertia_
            
            cluster_metrics[k] = {
                'silhouette_score': silhouette,
                'calinski_harabasz': calinski,
                'inertia': inertia,
                'model': kmeans
            }
        
        optimal_k = max(cluster_metrics.keys(), 
                       key=lambda k: cluster_metrics[k]['silhouette_score'])
        
        print(f"🎯 Optimal number of clusters: {optimal_k}")
        print(f"   Silhouette Score: {cluster_metrics[optimal_k]['silhouette_score']:.3f}")
        
        return cluster_metrics, optimal_k
    
    def advanced_segmentation(self, rfm_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Advanced customer segmentation with multiple algorithms"""
        
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'CustomerValue', 
                               'AvgOrderValue', 'PurchaseVelocity', 'PaymentRate']
        
        available_features = [f for f in clustering_features if f in rfm_data.columns]
        
        cluster_metrics, optimal_k = self.optimal_clustering(rfm_data, available_features)
        
        X = rfm_data[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # KMeans with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.config.RANDOM_STATE, n_init=10)
        rfm_data['segment_kmeans'] = kmeans.fit_predict(X_scaled)
        self.cluster_models['kmeans'] = kmeans
        
        # DBSCAN as alternative
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        rfm_data['segment_dbscan'] = dbscan.fit_predict(X_scaled)
        self.cluster_models['dbscan'] = dbscan
        
        metrics = {
            'kmeans_optimal_k': optimal_k,
            'kmeans_silhouette': silhouette_score(X_scaled, rfm_data['segment_kmeans']),
            'kmeans_calinski': calinski_harabasz_score(X_scaled, rfm_data['segment_kmeans']),
            'dbscan_clusters': len(set(rfm_data['segment_dbscan'])) - (1 if -1 in rfm_data['segment_dbscan'] else 0)
        }
        
        return rfm_data, metrics
