"""
Feature Selection and Dimensionality Reduction

Implements multiple feature selection strategies:
- PCA (Principal Component Analysis)
- Variance-based selection
- Correlation-based selection
- Feature importance from tree-based models
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select most important features using various methods."""

    def __init__(self, random_state: int = 42):
        """
        Initialize feature selector.

        Parameters
        ----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[pd.Series] = None

    def select_by_variance(
        self, X: pd.DataFrame, threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        threshold : float, default=0.01
            Variance threshold (0.0-1.0)

        Returns
        -------
        X_selected : pd.DataFrame
            Dataframe with selected features
        selected_features : List[str]
            Names of selected features
        """
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()

        logger.info(f"Variance selection: {X.shape[1]} -> {len(selected_features)} features")
        self.selected_features = selected_features
        return pd.DataFrame(X_selected, columns=selected_features), selected_features

    def select_by_correlation(
        self, X: pd.DataFrame, target: pd.Series, k: int = 20
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features most correlated with target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        target : pd.Series
            Target variable (RUL)
        k : int, default=20
            Number of features to select

        Returns
        -------
        X_selected : pd.DataFrame
            Dataframe with selected features
        selected_features : List[str]
            Names of selected features
        """
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, target)
        selected_features = X.columns[selector.get_support()].tolist()

        # Calculate correlation scores
        scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        logger.info(f"Correlation selection: {X.shape[1]} -> {len(selected_features)} features")
        logger.info(f"Top 5 features by f_regression score:\n{scores.head()}")

        self.selected_features = selected_features
        self.feature_importance = scores
        return pd.DataFrame(X_selected, columns=selected_features), selected_features

    def select_by_tree_importance(
        self, X: pd.DataFrame, target: pd.Series, k: int = 20, sample_frac: float = 0.3
    ) -> Tuple[pd.DataFrame, List[str], pd.Series]:
        """
        Select features by importance from Random Forest model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        target : pd.Series
            Target variable (RUL)
        k : int, default=20
            Number of features to select
        sample_frac : float, default=0.3
            Fraction of rows to sub-sample when len(X) > 50 000, for speed.

        Returns
        -------
        X_selected : pd.DataFrame
            Dataframe with top-k features
        selected_features : List[str]
            Names of selected features
        importances : pd.Series
            Feature importance values
        """
        # Sub-sample for speed on large datasets (e.g. 650k rows × 630 features)
        if len(X) > 20000:
            n_sample = min(20000, int(len(X) * sample_frac))
            idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
            X_fit = X.iloc[idx]
            y_fit = target.iloc[idx]
            logger.info(f"Tree importance: sub-sampled {len(X)} -> {len(X_fit)} rows")
        else:
            X_fit, y_fit = X, target

        # Reduced RF parameters to avoid multi-hour runtimes on CPU
        rf = RandomForestRegressor(
            n_estimators=10,   # minimal for importance ranking
            max_depth=6,       # shallow trees are faster
            n_jobs=1,          # single-threaded to avoid joblib deadlock in Jupyter
            random_state=self.random_state,
        )
        rf.fit(X_fit, y_fit)

        # Get importances
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        # Select top-k
        selected_features = importances.head(min(k, len(importances))).index.tolist()
        X_selected = X[selected_features]

        logger.info(f"Tree importance selection: {X.shape[1]} -> {len(selected_features)} features")
        logger.info(f"Top 10 features by importance:\n{importances.head(10)}")

        self.selected_features = selected_features
        self.feature_importance = importances
        return X_selected, selected_features, importances

    def select_by_pca(
        self, X: pd.DataFrame, n_components: Optional[int] = None, explained_variance: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        n_components : int, optional
            Number of components to keep. If None, use explained_variance threshold.
        explained_variance : float, default=0.95
            Cumulative explained variance ratio threshold (0-1)

        Returns
        -------
        X_transformed : np.ndarray
            PCA-transformed features
        pca : PCA
            Fitted PCA object (for transform on new data)
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PCA
        if n_components is None:
            pca = PCA(n_components=0.95)  # Auto-select based on variance
        else:
            pca = PCA(n_components=n_components)

        X_transformed = pca.fit_transform(X_scaled)

        logger.info(
            f"PCA: {X.shape[1]} -> {pca.n_components_} components "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})"
        )
        logger.info(f"PCA component variance ratios: {pca.explained_variance_ratio_[:5]}")

        self.feature_importance = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))])
        return X_transformed, pca

    def select_combined(
        self,
        X: pd.DataFrame,
        target: pd.Series,
        variance_threshold: float = 0.01,
        correlation_k: int = 30,
        tree_k: int = 20,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Combine multiple selection methods for robust feature selection.

        Process:
        1. Remove low-variance features
        2. Select top features by correlation
        3. Select top features by tree importance
        4. Return intersection (most robust)

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        target : pd.Series
            Target variable (RUL)
        variance_threshold : float, default=0.01
            Variance threshold
        correlation_k : int, default=30
            Number of features from correlation method
        tree_k : int, default=20
            Number of features from tree importance method

        Returns
        -------
        X_selected : pd.DataFrame
            Dataframe with combined-selected features
        selected_features : List[str]
            Names of selected features
        """
        # Step 1: Variance filtering
        X_var, feat_var = self.select_by_variance(X, variance_threshold)

        # Step 2: Correlation filtering
        _, feat_corr = self.select_by_correlation(X_var, target, k=correlation_k)

        # Step 3: Tree importance filtering
        _, feat_tree, _ = self.select_by_tree_importance(X_var, target, k=tree_k)

        # Step 4: Intersection (most robust features)
        combined_features = list(set(feat_corr) & set(feat_tree))
        combined_features = sorted(combined_features, key=lambda f: X.columns.tolist().index(f))

        X_selected = X[combined_features]

        logger.info(
            f"Combined selection: Variance({len(feat_var)}) ∩ Correlation({len(feat_corr)}) ∩ Tree({len(feat_tree)}) = {len(combined_features)} features"
        )
        logger.info(f"Selected features: {combined_features}")

        self.selected_features = combined_features
        return X_selected, combined_features

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary of feature importance.

        Returns
        -------
        summary : pd.DataFrame
            Feature importance summary
        """
        if self.feature_importance is None:
            logger.warning("No feature importance calculated yet")
            return pd.DataFrame()

        summary = pd.DataFrame({
            "feature": self.feature_importance.index,
            "importance": self.feature_importance.values,
            "importance_pct": (self.feature_importance.values / self.feature_importance.values.sum() * 100)
        })
        return summary.sort_values("importance", ascending=False)
