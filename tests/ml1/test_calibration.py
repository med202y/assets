"""Calibration chronology and fair comparison; never regenerate ML OOF output."""

from dataclasses import replace
import hashlib
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1 import calibration as cal
from src.models.ml1.core import DataValidationError
from src.models.ml1.time_validation import FEATURE_COLUMNS, LABEL, FINAL_TEST_START, WalkForwardConfig, prepare_development_data, build_walk_forward_folds


@pytest.fixture(scope="module")
def synthetic_inputs():
    sessions=xcals.get_calendar("ARCX",start="2013-01-01",end="2017-04-01").sessions
    dates=sessions[(sessions>="2013-01-03")&(sessions<"2017-01-01")]
    pos=sessions.get_indexer(dates)
    X=pd.DataFrame(np.random.default_rng(42).normal(size=(len(dates),10)),index=dates,columns=FEATURE_COLUMNS)
    y=pd.Series((np.arange(len(dates))%11==0).astype(int),index=dates,name=LABEL)
    info=pd.DataFrame({"target_start_session":sessions[pos+1],"target_end_session":sessions[pos+21],"future_session_count":21},index=dates)
    data=prepare_development_data(X,y,info,sessions)
    folds=build_walk_forward_folds(data,WalkForwardConfig(validation_years=(2015,2016)))
    index=data.X.index[np.concatenate([f.validation_indices for f in folds])]
    p=pd.Series(.1+.05*np.sin(np.arange(len(index))/12),index=index)
    return cal.CalibrationInputs(p,p*.9,data,folds,{})


def test_fixed_configuration_and_numerically_safe_logit():
    assert cal.calibration_spec("platt")=={"method":"platt","input":"logit_of_clipped_probability","epsilon":1e-6,"C":1.,"solver":"lbfgs","max_iter":2000,"class_weight":None}
    assert cal.calibration_spec("isotonic")=={"method":"isotonic","out_of_bounds":"clip","y_min":0.,"y_max":1.,"increasing":True}
    assert np.isfinite(cal.numeric_logit([0,1,.5])).all()
    assert cal.numeric_logit([.5])[0,0]==0


@pytest.mark.parametrize("n,pos,neg,expected",[(199,10,189,False),(200,10,190,True),(200,9,191,False),(200,191,9,False),(200,190,10,True)])
def test_support_boundaries_are_fixed(n,pos,neg,expected):
    s={"n_previous_oof":n,"n_calibration":n,"n_positive":pos,"n_negative":neg}
    assert (cal.support_status(s)=="calibrated")==expected


def test_2015_has_no_history_and_returns_exact_raw_probabilities(synthetic_inputs):
    x=synthetic_inputs;f=x.folds[0]
    p,history=cal.calibrate_fold(x.probabilities,x.data,f)
    assert p.calibration_status.eq("not_calibrated_no_history").all()
    assert p.n_calibration.eq(0).all()
    np.testing.assert_array_equal(p.predicted_probability,p.raw_probability)
    assert all(h["n_calibration"]==0 for h in history)


def test_history_uses_prior_year_and_excludes_exactly_21_immature_sessions(synthetic_inputs):
    x=synthetic_inputs;f=x.folds[1]
    dates,y,s=cal.admissible_history(x.probabilities,x.data,f)
    assert (dates.year<f.validation_year).all() and s["n_excluded_immature"]==21
    assert (x.data.target_info.loc[dates].target_end_session<f.validation_start).all()
    assert s["n_calibration"]==231 and len(y)==231


def test_immature_label_values_are_never_inspected(synthetic_inputs):
    class Forbidden:
        def __eq__(self,other):raise AssertionError("Immature label inspected")
    x=synthetic_inputs;f=x.folds[1]
    y=x.data.y.astype(object).copy()
    prior=x.probabilities.index[x.probabilities.index.year==2015]
    y.loc[prior[-21:]]=[Forbidden() for _ in range(21)]
    p,_=cal.calibrate_fold(x.probabilities,replace(x.data,y=y),f)
    assert p.calibration_status.eq("calibrated").all()


def test_current_validation_labels_do_not_change_calibrated_probabilities(synthetic_inputs):
    x=synthetic_inputs;f=x.folds[1]
    first,_=cal.calibrate_fold(x.probabilities,x.data,f)
    y=x.data.y.copy();y.iloc[f.validation_indices]=1-y.iloc[f.validation_indices]
    second,_=cal.calibrate_fold(x.probabilities,replace(x.data,y=y),f)
    np.testing.assert_array_equal(first.predicted_probability,second.predicted_probability)


@pytest.mark.parametrize("truth",[0,1])
def test_missing_class_falls_back_without_fitting(truth,synthetic_inputs,monkeypatch):
    x=synthetic_inputs;y=x.data.y.copy()
    y.loc[x.probabilities.index[x.probabilities.index.year==2015]]=truth
    def forbidden(*args,**kwargs):pytest.fail("Invalid calibrator fit")
    monkeypatch.setattr(cal,"LogisticRegression",forbidden);monkeypatch.setattr(cal,"IsotonicRegression",forbidden)
    p,_=cal.calibrate_fold(x.probabilities,replace(x.data,y=y),x.folds[1])
    assert p.calibration_status.eq("not_calibrated_insufficient_support").all()
    np.testing.assert_array_equal(p.predicted_probability,p.raw_probability)


def test_label_availability_check_is_independent_of_target_end(synthetic_inputs,monkeypatch):
    original=cal.decision_times
    monkeypatch.setattr(cal,"decision_times",lambda dates:original(dates)+pd.Timedelta(days=10000))
    x=synthetic_inputs
    dates,_,summary=cal.admissible_history(x.probabilities,x.data,x.folds[1])
    assert len(dates)==0 and summary["n_excluded_immature"]==252


@pytest.mark.parametrize("mutation",["2023","nan","inf","duplicate"])
def test_invalid_oof_inputs_are_rejected_before_calibrator_fit(mutation,synthetic_inputs):
    p=synthetic_inputs.probabilities.copy()
    if mutation=="2023":p.index=p.index+pd.DateOffset(years=10)
    elif mutation=="duplicate":p=pd.concat([p,p.iloc[:1]])
    else:p.iloc[0]=np.nan if mutation=="nan" else np.inf
    with pytest.raises(DataValidationError):cal.validate_oof_probabilities(p)


def test_methods_share_history_and_comparisons_share_dates(synthetic_inputs):
    r=cal.run_calibration(synthetic_inputs)
    for year in [2015,2016]:
        h=[x for x in r.history if x["validation_year"]==year]
        assert h[0]["history_dates_sha256"]==h[1]["history_dates_sha256"]
        assert h[0]["n_calibration"]==h[1]["n_calibration"]
    for scope,g in r.metrics.groupby("comparison_scope"):
        rows=g[g.period_type=="pooled_oof"]
        assert rows.n_observations.nunique()==1 and rows.n_positive.nunique()==1
    eligible=r.metrics[(r.metrics.comparison_scope=="actually_calibrated_common")&(r.metrics.period_type=="fold")]
    assert set(eligible.validation_year)=={2016}
    assert np.isfinite(r.predictions.predicted_probability).all()


def test_reproducibility_without_post_score_parameter_changes(synthetic_inputs):
    first=cal.run_calibration(synthetic_inputs);second=cal.run_calibration(synthetic_inputs)
    pd.testing.assert_frame_equal(first.predictions,second.predictions)
    assert first.provenance["no_calibration_method_selected"] is True
    assert first.provenance["base_production_refits"]==0


@pytest.mark.parametrize("mutation",["2023","probability","history"])
def test_export_safety_guards(mutation,synthetic_inputs):
    r=cal.run_calibration(synthetic_inputs);p=r.predictions.copy()
    if mutation=="2023":p.loc[0,"prediction_date"]=pd.Timestamp("2023-01-03")
    elif mutation=="probability":p.loc[0,"predicted_probability"]=1.1
    else:p.loc[p.calibration_status.eq("calibrated"),"calibration_target_end_max"]=pd.Timestamp("2022-01-01")
    with pytest.raises(DataValidationError):cal.validate_calibrated_predictions(p)


def test_existing_output_is_refused_before_any_plot_or_write(synthetic_inputs,tmp_path):
    folder=tmp_path/"existing";folder.mkdir();(folder/"keep").write_text("unchanged")
    result=cal.run_calibration(synthetic_inputs)
    with pytest.raises(FileExistsError):cal.export_results(result,None,synthetic_inputs.data,folder)
    assert (folder/"keep").read_text()=="unchanged"


def test_real_loading_and_calibration_preserve_sources_and_filter_final_labels(monkeypatch):
    root=cal.PROJECT_ROOT
    if not (root/"data/processed/ml1_training/training_metadata.json").exists():pytest.skip("Local OOF data absent")
    paths=list((root/"data/processed/ml1_training").iterdir())+list((root/"reports/ml1/evaluation").iterdir())
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    original=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if Path(path).name=="targets_ml1.parquet":
            assert kwargs.get("columns") is not None
            if LABEL in kwargs["columns"]:
                assert ("Date","<",FINAL_TEST_START) in kwargs.get("filters",[])
                assert ("target_end_session","<",FINAL_TEST_START) in kwargs.get("filters",[])
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    x=cal.load_calibration_inputs();r=cal.run_calibration(x)
    assert len(r.predictions)==4*1993
    assert len(r.metrics)==68  # 4*(8 years+pool) plus 4*(7 eligible years+pool).
    eligible=r.metrics[(r.metrics.comparison_scope=="actually_calibrated_common")&(r.metrics.period_type=="pooled_oof")]
    assert eligible.n_observations.eq(1741).all()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
