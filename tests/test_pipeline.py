"""Regression checks for data boundaries and training-only preprocessing."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import split_application_ids
from src.evaluation import evaluation_summary
from src.features import add_ratio_features, add_employment_features
from src.modeling import WOETransformer, NativeCategoricalTransformer, model_feature_columns


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "flag": np.repeat([0, 1], 100),
            "category": ["a"] * 5 + [None] * 10 + ["a"] * 85 + ["b"] * 100,
        }, index=np.arange(1000, 1200))
        self.y = pd.Series([1] * 10 + [0] * 90 + [1] * 40 + [0] * 60, index=self.X.index)

    def test_binary_iv_missing_and_unknown(self):
        encoder = WOETransformer().fit(self.X, self.y)
        iv = encoder.iv_summary_.set_index("name").loc["flag", "iv"]
        self.assertGreater(iv, 0.02)
        self.assertIn("flag", encoder.categorical_features_)
        query = pd.DataFrame({"flag": [0, 1], "category": [None, "new"]})
        self.assertTrue(np.isfinite(encoder.transform(query).to_numpy()).all())
        all_values = encoder._transform_all(encoder._prepare(query))
        self.assertEqual(all_values.loc[1, "category"], 0.0)
        table = encoder.binning_process_.get_binned_variable("category").binning_table.build()
        empirical_missing = table.loc[table["Bin"].eq("Missing"), "WoE"].iloc[0]
        self.assertGreater(abs(float(empirical_missing)), 0.01)
        self.assertAlmostEqual(float(all_values.loc[0, "category"]), float(empirical_missing))

    def test_fold_isolation_and_roundtrip(self):
        for tr, va in StratifiedKFold(2, shuffle=True, random_state=42).split(self.X, self.y):
            encoder = WOETransformer().fit(self.X.iloc[tr], self.y.iloc[tr])
            self.assertEqual(set(encoder.fit_index_), set(self.X.iloc[tr].index))
            self.assertTrue(set(encoder.fit_index_).isdisjoint(self.X.iloc[va].index))
            encoder.transform(self.X.iloc[va])
        pipeline = Pipeline([("preprocess", WOETransformer()), ("model", LogisticRegression())])
        pipeline.fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(pipeline, path)
            np.testing.assert_allclose(pipeline.predict_proba(self.X), joblib.load(path).predict_proba(self.X))

    def test_native_unknown_and_column_order(self):
        encoder = NativeCategoricalTransformer().fit(self.X)
        query = pd.DataFrame({"category": ["new", None], "flag": [1, 0]})
        encoded = encoder.transform(query)
        self.assertEqual(list(encoded), list(self.X))
        self.assertTrue(encoded["category"].isna().all())
        self.assertNotIn("new", encoder.categories_["category"])

    def test_ratios_and_employment(self):
        frame = pd.DataFrame({
            "AMT_CREDIT": [100., 100., 100., np.inf], "AMT_INCOME_TOTAL": [0., -1., np.nan, 1.],
            "AMT_ANNUITY": [0., 10., 10., 10.], "DAYS_BIRTH": [0, -3650, -3650, -3650],
            "DAYS_EMPLOYED": [365243, -365, -365, -365],
        })
        derived = add_employment_features(add_ratio_features(frame))
        for col in ["DEBT_TO_INCOME_RATIO", "ANNUITY_TO_INCOME_RATIO", "CREDIT_TO_ANNUITY_RATIO", "EMPLOYMENT_TO_AGE_RATIO"]:
            self.assertFalse(np.isinf(derived[col]).any())
        self.assertTrue(derived["DEBT_TO_INCOME_RATIO"].isna().all())
        self.assertTrue(np.isnan(derived.loc[0, "EMPLOYMENT_TO_AGE_RATIO"]))

    def test_split_and_trust_boundaries(self):
        frame = pd.DataFrame({"SK_ID_CURR": np.arange(200), "TARGET": self.y.to_numpy()})
        pd.testing.assert_frame_equal(split_application_ids(frame), split_application_ids(frame.sample(frac=1, random_state=7)))
        manifest = split_application_ids(frame)
        self.assertEqual(manifest["split"].value_counts().to_dict(), {"train": 160, "test": 40})
        with self.assertRaises(ValueError):
            split_application_ids(pd.concat([frame, frame.iloc[:1]]))
        with self.assertRaises(ValueError):
            NativeCategoricalTransformer().fit(frame)
        self.assertEqual(model_feature_columns(frame.assign(DAYS_BIRTH=-1, DAYS_EMPLOYED=-1, age=1)), ["age"])

    def test_unrounded_metrics_and_invalid_probabilities(self):
        metrics = evaluation_summary([0, 0, 1, 1], [0.1, 0.7, 0.2, 0.9])
        self.assertEqual(set(metrics.index), {"AUC", "Gini", "KS", "AP", "Brier", "LogLoss"})
        self.assertAlmostEqual(metrics["Gini"], 2 * metrics["AUC"] - 1)
        unrounded = evaluation_summary([0, 0, 0, 1, 1, 1], [.1, .3, .6, .2, .4, .9])
        self.assertNotEqual(unrounded["AUC"], round(unrounded["AUC"], 4))
        with self.assertRaises(ValueError):
            evaluation_summary([0, 1], [np.nan, .7])
        with self.assertRaises(ValueError):
            evaluation_summary([0, 1], [-.1, .7])


if __name__ == "__main__":
    unittest.main()
