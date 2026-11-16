"""Integration tests use a completely synthetic price dataset and frozen fit."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1 import inference as inf
from src.models.ml1 import train_ml1 as train
from src.models.ml1.core import DataValidationError, build_features
from src.models.ml1.pipeline import build_dataset
from src.models.ml1.time_validation import FEATURE_COLUMNS, FINAL_TEST_START, LABEL, load_development_data


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root=tmp_path_factory.mktemp("ml1-inference-synthetic")
    raw=root/"raw";raw.mkdir()
    sessions=xcals.get_calendar("ARCX",start="2004-09-01",end="2024-03-10").sessions
    sessions=sessions[sessions<="2024-03-06"]
    rng=np.random.default_rng(42)
    increments=rng.normal(.0001,.012,len(sessions))
    increments[np.arange(len(sessions))%211==0]=-.1
    spy=100*np.exp(np.cumsum(increments))
    ief=100*np.exp(np.cumsum(rng.normal(.00005,.004,len(sessions))))
    for ticker,values in [("SPY",spy),("IEF",ief)]:
        pd.DataFrame({"Date":sessions.strftime("%Y-%m-%d"),"Adj Close":values}).to_csv(raw/f"{ticker}_raw.csv",index=False)
    dataset=root/"dataset";build_dataset(raw,dataset)
    data=load_development_data(dataset,raw/"SPY_raw.csv")
    # Fabricate synthetic OOF probabilities via spies; no real RF/XGB/24-model refit.
    class Spy:
        classes_=np.array([0,1])
        def fit(self,X,y):return self
        def predict_proba(self,X):return np.tile([.9,.1],(len(X),1))
        def get_params(self,deep=True):return {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train,"build_estimator",lambda name:Spy())
        patch.setattr(train,"xgboost_preflight",lambda:{"status":"passed"})
        run=train.train_walk_forward(data)
        training=root/"training";train.export_training(run,training)
    release=root/"release"
    manifest=inf.create_release(release,dataset_dir=dataset,raw_dir=raw,training_dir=training,expected_rows=len(data.X))
    predictor=inf.load_inference(release,trusted=True)
    return root,raw,dataset,training,release,data,manifest,predictor


def test_policy_is_provisional_raw_core():
    assert inf.POLICY=={"model":"logistic_regression","feature_variant":"core","calibration":"none","status":"provisional","horizon":"21_spy_sessions"}
    contract=inf.integration_contract()
    assert contract["horizon_sessions"]==21 and "0.93" in contract["meaning"]


@pytest.mark.parametrize("value",[None,42,"wrong","2022-02-30","2022/01/03","2022-01-03T12:00:00",pd.Timestamp("2022-01-03",tz="UTC")])
def test_invalid_dates_are_rejected(value):
    with pytest.raises(DataValidationError):inf.parse_prediction_date(value)


def test_trusted_loading_is_explicit_and_missing_release_is_clear(tmp_path):
    with pytest.raises(PermissionError):inf.load_inference(tmp_path)
    with pytest.raises(FileNotFoundError):inf.load_inference(tmp_path,trusted=True)


def test_simple_interface_requires_configuration_and_returns_python_float(workspace,monkeypatch):
    _,_,_,_,release,_,_,_=workspace
    monkeypatch.setattr(inf,"_DEFAULT_PREDICTOR",None)
    with pytest.raises(RuntimeError,match="configure_inference"):inf.predict_crisis("2022-11-30")
    inf.configure_inference(release,trusted=True)
    p=inf.predict_crisis("2022-11-30")
    assert type(p) is float and 0<=p<=1
    d=inf.predict_crisis_details("2022-11-30")
    assert d["p_crisis"]==p and d["prediction_origin"]=="historical_oof"


def test_historical_predictions_never_use_frozen_model(workspace):
    *_,predictor=workspace
    def forbidden(*args,**kwargs):pytest.fail("Frozen estimator must not predict historical OOF")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(predictor.model,"predict_proba",forbidden)
        date=predictor.oof.index[-1]
        assert predictor.predict_crisis(date)==float(predictor.oof.loc[date].predicted_probability)
        detail=predictor.predict_details(date)
        assert detail.training_cutoff=="2021-12-01" and detail.training_label_end=="2021-12-31"


@pytest.mark.parametrize("date",["2010-01-04","2022-12-01","2022-11-26","2024-03-07","2099-01-01"])
def test_in_sample_unavailable_calendar_and_future_dates_refused(date,workspace):
    *_,predictor=workspace
    with pytest.raises(DataValidationError):predictor.predict_crisis(date)


def test_blind_inference_does_not_read_any_targets(workspace,monkeypatch):
    *_,predictor=workspace
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        assert "targets" not in Path(path).name
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    detail=predictor.predict_details("2023-01-31")
    assert type(detail.p_crisis) is float and 0<=detail.p_crisis<=1
    assert detail.prediction_origin=="blind_holdout_inference" and not detail.performance_evaluated
    assert detail.calibration_method=="none" and detail.status=="provisional"
    assert "17:00:00" in detail.decision_time


def test_scaler_is_fitted_only_on_admissible_development(workspace):
    _,_,_,_,_,data,manifest,predictor=workspace
    scaler=predictor.model.named_steps["scaler"]
    assert scaler.n_samples_seen_==len(data.X)==manifest["training_rows"]
    np.testing.assert_allclose(scaler.mean_,data.X.mean(axis=0))
    np.testing.assert_allclose(scaler.var_,data.X.var(axis=0,ddof=0))
    assert data.X.index.max()<FINAL_TEST_START and data.target_info.target_end_session.max()<FINAL_TEST_START
    assert predictor.model.named_steps["classifier"].get_params()["C"]==1.


def test_feature_prefix_never_passes_later_prices_to_transform(workspace,monkeypatch):
    *_,predictor=workspace
    original=inf.build_features;seen=[];cutoff=pd.Timestamp("2023-01-31")
    def guarded(spy,ief):
        assert spy.index.max()==cutoff and ief.index.max()<=cutoff
        seen.append(True)
        return original(spy,ief)
    monkeypatch.setattr(inf,"build_features",guarded)
    predictor.predict_crisis(cutoff)
    assert seen==[True]


def test_feature_values_are_invariant_to_later_prices_in_memory(workspace):
    *_,predictor=workspace;source=predictor.feature_source;cutoff=pd.Timestamp("2023-01-31")
    first=source.feature_at(cutoff)
    original_spy,original_ief=source.spy,source.ief
    try:
        source.spy=source.spy.copy();source.ief=source.ief.copy()
        source.spy.loc[source.spy.index>cutoff]*=1000
        source.ief.loc[source.ief.index>cutoff]*=.001
        pd.testing.assert_frame_equal(first,source.feature_at(cutoff))
    finally:source.spy,source.ief=original_spy,original_ief


@pytest.mark.parametrize("mutation",["nan","wrong_value","extra_column","order"])
def test_missing_future_or_reordered_features_cannot_be_used(mutation,workspace):
    *_,predictor=workspace;source=predictor.feature_source;original=source.features
    source.features=source.features.copy()
    try:
        if mutation=="nan":source.features.loc[pd.Timestamp("2023-01-31"),FEATURE_COLUMNS[0]]=np.nan
        elif mutation=="wrong_value":source.features.loc[pd.Timestamp("2023-01-31"),FEATURE_COLUMNS[0]]+=1
        elif mutation=="extra_column":source.features["VIX_level"]=1.
        else:source.features=source.features.iloc[:,::-1]
        with pytest.raises(DataValidationError):source.feature_at("2023-01-31")
    finally:source.features=original


def test_monthly_oof_has_95_months_and_never_invents_december_2022(workspace):
    _,_,_,_,release,_,_,predictor=workspace
    table=inf.monthly_oof_probabilities(predictor)
    assert len(table)==95 and table.month.iloc[0]=="2015-01" and table.month.iloc[-1]=="2022-11"
    assert not table.month.duplicated().any() and table.prediction_origin.eq("historical_oof").all()
    assert not {"observed_target",LABEL,"y","future_min_price_21"}.intersection(table.columns)
    assert pd.to_datetime(table.training_label_end).lt(FINAL_TEST_START).all()
    saved=pd.read_csv(release/"monthly_oof_probabilities.csv")
    np.testing.assert_allclose(saved.p_crisis,table.p_crisis,atol=1e-15,rtol=0)
    import csv
    with (release/"monthly_oof_probabilities.csv").open() as f:
        exact=[float(row["p_crisis"]) for row in csv.DictReader(f)]
    np.testing.assert_array_equal(exact,table.p_crisis)
    for row in table.itertuples():assert row.p_crisis==predictor.predict_crisis(row.prediction_date)


def test_monthly_blind_excludes_partial_month_and_separates_origins(workspace):
    _,_,_,_,release,_,_,predictor=workspace
    table=inf.monthly_blind_probabilities(predictor)
    assert len(table)==14 and table.month.iloc[-1]=="2024-02"  # March has only four sessions locally.
    assert table.prediction_origin.eq("blind_holdout_inference").all()
    assert not table.performance_evaluated.any()
    assert table.prediction_date.min()>="2023-01-01"
    assert len(pd.read_csv(release/"monthly_blind_probabilities.csv"))==14


def test_incomplete_month_with_missing_feature_session_is_not_exported(workspace):
    *_,predictor=workspace;source=predictor.feature_source;original=source.features
    try:
        source.features=original.drop(pd.Timestamp("2023-01-10"))
        table=inf.monthly_blind_probabilities(predictor)
        assert "2023-01" not in set(table.month)
    finally:source.features=original


def test_frozen_fit_reproducibility_on_synthetic_data(workspace):
    _,_,_,_,_,data,_,predictor=workspace
    second,_=inf.fit_frozen_model(data,expected_rows=len(data.X))
    rows=predictor.feature_source.features.loc["2023-01-03":"2023-01-10"]
    np.testing.assert_array_equal(second.predict_proba(rows),predictor.model.predict_proba(rows))


def test_frozen_training_rejects_changed_row_count_and_future_target(workspace):
    _,_,_,_,_,data,_,_=workspace
    with pytest.raises(DataValidationError,match="row count"):inf.fit_frozen_model(data,expected_rows=1)
    info=data.target_info.copy();info.iloc[-1,info.columns.get_loc("target_end_session")]=pd.Timestamp("2023-01-03")
    with pytest.raises(DataValidationError,match="pre-2023"):inf.fit_frozen_model(replace(data,target_info=info),expected_rows=len(data.X))


@pytest.mark.parametrize("mutation",["changed","missing"])
def test_artifacts_model_and_manifest_are_checked_before_unpickling(mutation,workspace,tmp_path,monkeypatch):
    import shutil
    _,_,_,_,release,_,_,_=workspace
    copied=tmp_path/"copy";shutil.copytree(release,copied)
    if mutation=="changed":
        with (copied/"logistic_regression_final.joblib").open("ab") as f:f.write(b'changed')
    else:(copied/"logistic_regression_final.joblib").unlink()
    monkeypatch.setattr(inf.joblib,"load",lambda *args:pytest.fail("Must verify hashes before loading"))
    with pytest.raises((DataValidationError,FileNotFoundError)):inf.load_inference(copied,trusted=True)


def test_no_overwrite_and_no_fit_when_destination_exists(workspace,monkeypatch):
    _,_,_,_,release,_,_,_=workspace
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in release.iterdir()}
    monkeypatch.setattr(inf,"fit_frozen_model",lambda *args,**kwargs:pytest.fail("No fit before overwrite refusal"))
    with pytest.raises(FileExistsError):inf.create_release(release)
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in release.iterdir()}


def test_live_loading_and_feature_reader_never_load_labels(workspace,monkeypatch):
    _,_,_,_,release,_,_,_=workspace
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        assert "targets" not in Path(path).name
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    engine=inf.load_inference(release,trusted=True)
    assert 0<=engine.predict_crisis("2023-01-31")<=1


def test_real_data_audit_filters_final_labels_without_any_frozen_fit(monkeypatch):
    if not (inf.DEFAULT_TRAINING/"training_metadata.json").exists():pytest.skip("Local artifacts absent")
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if Path(path).name=="targets_ml1.parquet":
            assert kwargs.get("columns") is not None
            if LABEL in kwargs["columns"]:
                assert ("Date","<",FINAL_TEST_START) in kwargs.get("filters",[])
                assert ("target_end_session","<",FINAL_TEST_START) in kwargs.get("filters",[])
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    data=load_development_data();oof,_=inf.load_saved_oof(data)
    assert len(data.X)==4329 and len(oof)==1993
    assert data.target_info.target_end_session.max()==pd.Timestamp("2022-12-30")
    source=inf.FeatureSource()
    row=source.feature_at("2023-01-31")
    assert tuple(row.columns)==FEATURE_COLUMNS
