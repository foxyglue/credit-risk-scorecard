"""Train-fitted preprocessing shared by CV, final fits, and saved inference."""

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import ID_COL, TARGET_COL


def model_feature_columns(frame):
    """Keep raw columns in the master, but model their cleaned equivalents."""
    excluded = {ID_COL, TARGET_COL, "DAYS_BIRTH", "DAYS_EMPLOYED"}
    return [c for c in frame.columns if c not in excluded]


def _validate_input(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("A DataFrame with unique feature names is required.")
    if {ID_COL, TARGET_COL}.intersection(frame.columns):
        raise ValueError("IDs and TARGET cannot be predictors.")
    if np.isinf(frame.select_dtypes(include="number").to_numpy(dtype=float)).any():
        raise ValueError("Features contain infinity; fix their construction before fitting.")


class WOETransformer(TransformerMixin, BaseEstimator):
    """Optimal binning, train-only IV screening, and WOE redundancy removal."""

    def __init__(self, iv_min=0.02, correlation_limit=0.90):
        self.iv_min = iv_min
        self.correlation_limit = correlation_limit

    def _prepare(self, frame):
        prepared = frame.loc[:, self.feature_names_in_].copy()
        for col in self.categorical_features_:
            prepared[col] = prepared[col].astype(object).where(prepared[col].notna(), np.nan)
        return prepared

    def fit(self, X, y):
        _validate_input(X)
        self.feature_names_in_ = np.array([c for c in X if X[c].nunique(dropna=False) > 1])
        if not len(self.feature_names_in_):
            raise ValueError("No nonconstant features are available for binning.")
        self.n_features_in_ = X.shape[1]
        self.fit_index_ = X.index.to_numpy(copy=True)
        self.categorical_features_ = [c for c in self.feature_names_in_ if
            not pd.api.types.is_numeric_dtype(X[c]) or
            set(X[c].dropna().unique()).issubset({0, 1})]
        self.categories_ = {c: X[c].dropna().unique().tolist() for c in self.categorical_features_}
        numerical = set(self.feature_names_in_) - set(self.categorical_features_)
        self.binning_process_ = BinningProcess(
            variable_names=self.feature_names_in_.tolist(),
            categorical_variables=self.categorical_features_,
            max_n_bins=5, min_bin_size=0.05,
            binning_fit_params={c: {"monotonic_trend": "auto_asc_desc"} for c in numerical},
        )
        prepared = self._prepare(X)
        self.binning_process_.fit(prepared, np.asarray(y))
        self.iv_summary_ = self.binning_process_.summary().sort_values(
            ["iv", "name"], ascending=[False, True], kind="stable",
        ).reset_index(drop=True)
        candidates = self.iv_summary_.loc[
            self.iv_summary_["iv"].astype(float).ge(self.iv_min), "name"
        ].tolist()
        if not candidates:
            raise ValueError("No features passed the training IV screen.")
        transformed = self._transform_all(prepared)[candidates]
        correlations = transformed.corr().abs()
        self.selected_features_ = []
        self.redundant_features_ = {}
        for col in candidates:
            matches = [kept for kept in self.selected_features_
                       if correlations.loc[col, kept] >= self.correlation_limit]
            if matches:
                self.redundant_features_[col] = matches[0]
            else:
                self.selected_features_.append(col)
        self.iv_summary_["selected"] = self.iv_summary_["name"].isin(self.selected_features_)
        return self

    def _transform_all(self, prepared):
        transformed = self.binning_process_.transform(
            prepared, metric="woe", metric_missing="empirical", metric_special="empirical",
        )
        return pd.DataFrame(np.asarray(transformed, dtype=float),
                            columns=self.feature_names_in_, index=prepared.index)

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        _validate_input(X)
        result = self._transform_all(self._prepare(X))[self.selected_features_]
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("WOE transform produced nonfinite values; inspect the binning tables.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.array(self.selected_features_)


class NativeCategoricalTransformer(TransformerMixin, BaseEstimator):
    """Keep numerical NaNs; freeze categorical levels using training data only."""

    def fit(self, X, y=None):
        _validate_input(X)
        self.feature_names_in_ = np.array([c for c in X if X[c].nunique(dropna=False) > 1])
        self.n_features_in_ = X.shape[1]
        self.fit_index_ = X.index.to_numpy(copy=True)
        self.categories_ = {c: X[c].dropna().unique().tolist() for c in self.feature_names_in_
                            if not pd.api.types.is_numeric_dtype(X[c])}
        return self

    def transform(self, X):
        check_is_fitted(self, "categories_")
        _validate_input(X)
        result = X.loc[:, self.feature_names_in_].copy()
        for col, levels in self.categories_.items():
            known = result[col].where(result[col].isin(levels), np.nan)
            result[col] = pd.Categorical(known, categories=levels)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "categories_")
        return self.feature_names_in_.copy()
