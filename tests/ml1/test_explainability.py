"""SHAP tests use synthetic historical data, never a final production estimator."""

from dataclasses import replace
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1 import explain_ml1 as explain
from src.models.ml1.calibration import plotting_environment
from src.models.ml1.core import DataValidationError
from src.models.ml1.time_validation import FEATURE_COLUMNS,LABEL,WalkForwardConfig,prepare_development_data,build_walk_forward_folds
from src.models.ml1.train_ml1 import fit_predict_fold


@pytest.fixture(scope="module")
def fitted():
    sessions=xcals.get_calendar("ARCX",start="2013-01-01",end="2016-03-31").sessions
    dates=sessions[(sessions>="2013-01-03")&(sessions<"2016-01-01")]
    pos=sessions.get_indexer(dates)
    X=pd.DataFrame(np.random.default_rng(42).normal(size=(len(dates),10)),index=dates,columns=FEATURE_COLUMNS)
    y=pd.Series((np.arange(len(dates))%11==0).astype(int),index=dates,name=LABEL)
    info=pd.DataFrame({"target_start_session":sessions[pos+1],"target_end_session":sessions[pos+21],"future_session_count":21},index=dates)
    data=prepare_development_data(X,y,info,sessions)
    fold=build_walk_forward_folds(data,WalkForwardConfig(validation_years=(2015,)))[0]
    fit=fit_predict_fold("logistic_regression",data,fold)
    return data,fold,fit


@pytest.fixture(scope="module")
def explanation(fitted):
    data,fold,fit=fitted
    return explain.explain_fitted_pipeline(fit.estimator,data.X.iloc[fold.train_indices],data.X.iloc[fold.validation_indices])


def test_shap_is_actual_linear_explainer_with_train_only_background(fitted,monkeypatch):
    data,fold,fit=fitted
    train=data.X.iloc[fold.train_indices];validation=data.X.iloc[fold.validation_indices]
    with plotting_environment():
        import shap
    original=shap.LinearExplainer;called=[]
    def guarded(model,masker,*args,**kwargs):
        np.testing.assert_allclose(masker.data,fit.estimator.named_steps["scaler"].transform(train))
        assert len(masker.data)==len(train)
        called.append(True)
        return original(model,masker,*args,**kwargs)
    monkeypatch.setattr(shap,"LinearExplainer",guarded)
    result=explain.explain_fitted_pipeline(fit.estimator,train,validation)
    assert called==[True] and result.metadata["background_from_train_only"] is True
    assert result.metadata["background_end"]<result.metadata["validation_start"]


def test_shap_additivity_log_odds_and_importance(explanation,fitted):
    data,fold,fit=fitted;result=explanation
    validation=data.X.iloc[fold.validation_indices]
    np.testing.assert_allclose(result.base_values+result.values.sum(axis=1),fit.estimator.decision_function(validation),atol=1e-9)
    assert result.values.shape==(len(validation),10)
    assert "log-odds" in result.metadata["output_scale"]
    assert set(result.importance.feature)==set(FEATURE_COLUMNS)
    assert result.importance.mean_abs_shap_log_odds.is_monotonic_decreasing


def test_explanations_are_reproducible(explanation,fitted):
    data,fold,fit=fitted
    second=explain.explain_fitted_pipeline(fit.estimator,data.X.iloc[fold.train_indices],data.X.iloc[fold.validation_indices])
    np.testing.assert_array_equal(explanation.values,second.values)
    pd.testing.assert_frame_equal(explanation.importance,second.importance)


def test_reconstruction_matches_saved_synthetic_oof(fitted):
    data,fold,fit=fitted
    p=pd.Series(fit.probabilities,index=data.X.index[fold.validation_indices])
    result=explain.reconstruct_and_explain(data,fold,p)
    assert result.metadata["probability_max_abs_error"]<=1e-10
    assert result.metadata["validation_year"]==2015


def test_divergence_stops_before_any_shap_interpretation(fitted,monkeypatch):
    data,fold,fit=fitted
    p=pd.Series(np.clip(fit.probabilities+.01,0,1),index=data.X.index[fold.validation_indices])
    monkeypatch.setattr(explain,"explain_fitted_pipeline",lambda *args:pytest.fail("SHAP must not run after divergence"))
    with pytest.raises(DataValidationError,match="diverges"):
        explain.reconstruct_and_explain(data,fold,p)


@pytest.mark.parametrize("mutation",["extra","reordered","2023","nan"])
def test_invalid_features_or_final_dates_cannot_be_explained(mutation,fitted):
    data,fold,fit=fitted;train=data.X.iloc[fold.train_indices].copy();val=data.X.iloc[fold.validation_indices].copy()
    if mutation=="extra":val["future_min_price_21"]=1.
    elif mutation=="reordered":val=val.iloc[:,::-1]
    elif mutation=="2023":val.index=val.index+pd.DateOffset(years=10)
    else:val.iloc[0,0]=np.nan
    with pytest.raises(DataValidationError):explain.explain_fitted_pipeline(fit.estimator,train,val)


def test_scaler_fitted_on_wrong_data_is_rejected(fitted):
    from sklearn.base import clone
    data,fold,fit=fitted;model=clone(fit.estimator)
    model.fit(data.X,data.y.to_numpy(dtype=int))
    with pytest.raises(DataValidationError,match="Scaler"):
        explain.explain_fitted_pipeline(model,data.X.iloc[fold.train_indices],data.X.iloc[fold.validation_indices])


def test_shap_export_creates_three_outputs_and_refuses_overwrite(explanation,tmp_path):
    explain.export_explanation(explanation,tmp_path)
    assert {p.name for p in tmp_path.iterdir()}=={"shap_summary.png","shap_importance.png","shap_feature_importance.csv"}
    assert len(pd.read_csv(tmp_path/"shap_feature_importance.csv"))==10
    with pytest.raises(FileExistsError):explain.export_explanation(explanation,tmp_path)


def test_explanations_do_not_depend_on_validation_labels(fitted):
    data,fold,fit=fitted;y=data.y.copy();y.iloc[fold.validation_indices]=1-y.iloc[fold.validation_indices]
    p=pd.Series(fit.probabilities,index=data.X.index[fold.validation_indices])
    a=explain.reconstruct_and_explain(data,fold,p)
    b=explain.reconstruct_and_explain(replace(data,y=y),fold,p)
    np.testing.assert_array_equal(a.values,b.values)
