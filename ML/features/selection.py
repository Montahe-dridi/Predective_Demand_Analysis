# ML/features/selection.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FeatureSelector:
    """Advanced feature selection methods"""
    
    def __init__(self):
        self.encoders = {}
        self.feature_scores = {}
        self.selected_features = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       task_type: str = 'regression', 
                       method: str = 'statistical',
                       k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligent feature selection based on various methods"""
        
        # Prepare data
        X_processed = self._prepare_features_for_selection(X, y)
        
        if method == 'statistical':
            return self._statistical_selection(X_processed, y, task_type, k)
        elif method == 'mutual_info':
            return self._mutual_info_selection(X_processed, y, task_type, k)
        elif method == 'correlation':
            return self._correlation_selection(X_processed, y, task_type, k)
        elif method == 'variance':
            return self._variance_selection(X_processed, y, k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _prepare_features_for_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Prepare features for selection by encoding categorical variables"""
        X_processed = X.copy()
        
        # Separate numeric and categorical features
        numeric_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Encode categorical features
        for cat_col in categorical_features:
            if cat_col not in self.encoders:
                self.encoders[cat_col] = LabelEncoder()
                try:
                    X_processed[cat_col] = self.encoders[cat_col].fit_transform(X_processed[cat_col].astype(str))
                except Exception as e:
                    print(f"⚠️ Error encoding {cat_col}: {e}")
                    X_processed[cat_col] = 0
            else:
                try:
                    X_processed[cat_col] = self.encoders[cat_col].transform(X_processed[cat_col].astype(str))
                except Exception as e:
                    print(f"⚠️ Error transforming {cat_col}: {e}")
                    X_processed[cat_col] = 0
        
        # Handle any remaining non-numeric columns
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
        
        return X_processed
    
    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series, 
                             task_type: str, k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Statistical feature selection using F-tests"""
        
        if task_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False)
        
        self.feature_scores[f"{task_type}_statistical"] = feature_scores
        self.selected_features[f"{task_type}_statistical"] = selected_features
        
        print(f"📊 Statistical selection: {len(selected_features)} features selected")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series,
                             task_type: str, k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Mutual information based feature selection"""
        
        if task_type == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Select top k features
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        selected_features = feature_scores.head(k)['feature'].tolist()
        X_selected = X[selected_features]
        
        self.feature_scores[f"{task_type}_mutual_info"] = feature_scores
        self.selected_features[f"{task_type}_mutual_info"] = selected_features
        
        print(f"📊 Mutual info selection: {len(selected_features)} features selected")
        return X_selected, selected_features
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series,
                             task_type: str, k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Correlation-based feature selection"""
        
        # Calculate correlations with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Remove highly correlated features among themselves
        selected_features = []
        correlation_matrix = X.corr()
        
        for feature in correlations.index:
            if len(selected_features) >= k:
                break
                
            # Check correlation with already selected features
            if not selected_features:
                selected_features.append(feature)
            else:
                max_corr = correlation_matrix.loc[feature, selected_features].abs().max()
                if max_corr < 0.8:  # Threshold for multicollinearity
                    selected_features.append(feature)
        
        X_selected = X[selected_features]
        
        feature_scores = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        self.feature_scores[f"{task_type}_correlation"] = feature_scores
        self.selected_features[f"{task_type}_correlation"] = selected_features
        
        print(f"📊 Correlation selection: {len(selected_features)} features selected")
        return X_selected, selected_features
    
    def _variance_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Variance-based feature selection (remove low variance features)"""
        
        from sklearn.feature_selection import VarianceThreshold
        
        # Remove features with very low variance
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_selected = variance_selector.fit_transform(X)
        
        remaining_features = X.columns[variance_selector.get_support()].tolist()
        
        # Then select top k by variance
        variances = X[remaining_features].var().sort_values(ascending=False)
        selected_features = variances.head(k).index.tolist()
        
        X_selected = X[selected_features]
        
        feature_scores = pd.DataFrame({
            'feature': variances.index,
            'variance': variances.values
        })
        
        self.feature_scores["variance"] = feature_scores
        self.selected_features["variance"] = selected_features
        
        print(f"📊 Variance selection: {len(selected_features)} features selected")
        return X_selected, selected_features
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                 task_type: str = 'regression', 
                                 methods: List[str] = None,
                                 k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        """Ensemble feature selection combining multiple methods"""
        
        if methods is None:
            methods = ['statistical', 'mutual_info', 'correlation']
        
        all_selected_features = {}
        feature_votes = {}
        
        # Run each selection method
        for method in methods:
            try:
                _, features = self.select_features(X, y, task_type, method, k)
                all_selected_features[method] = features
                
                # Count votes for each feature
                for i, feature in enumerate(features):
                    if feature not in feature_votes:
                        feature_votes[feature] = 0
                    # Weight by rank (higher rank = more votes)
                    feature_votes[feature] += (k - i) / k
                    
            except Exception as e:
                print(f"⚠️ {method} selection failed: {e}")
                continue
        
        # Select features with highest votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        final_selected = [feat for feat, votes in sorted_features[:k]]
        
        X_final = X[final_selected]
        
        ensemble_scores = pd.DataFrame(sorted_features, columns=['feature', 'ensemble_score'])
        self.feature_scores[f"{task_type}_ensemble"] = ensemble_scores
        self.selected_features[f"{task_type}_ensemble"] = final_selected
        
        print(f"📊 Ensemble selection: {len(final_selected)} features selected")
        print(f"🗳️ Feature voting results saved")
        
        return X_final, final_selected
    
    def get_feature_selection_report(self) -> pd.DataFrame:
        """Generate comprehensive feature selection report"""
        
        report_data = []
        
        for method_name, scores_df in self.feature_scores.items():
            selected_features = self.selected_features.get(method_name, [])
            
            for _, row in scores_df.iterrows():
                report_data.append({
                    'method': method_name,
                    'feature': row['feature'],
                    'score': row.get('score', row.get('mi_score', row.get('correlation', row.get('variance', 0)))),
                    'selected': row['feature'] in selected_features,
                    'rank': list(scores_df['feature']).index(row['feature']) + 1
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Calculate feature selection frequency
        feature_frequency = report_df[report_df['selected']].groupby('feature').size()
        feature_frequency.name = 'selection_frequency'
        
        return report_df, feature_frequency