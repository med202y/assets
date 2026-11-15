"""Baseline-only tests: no classifier training and no final-test outcome reads."""

from dataclasses import replace
import hashlib
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.ml1.baselines import HistoricalFrequencyBaseline, VixThresholdBaseline
from src.models.ml1.core import DataValidationError
from src.models.ml1.evaluate_baselines import (
    RESULT_FILES, evaluate_baselines, evaluate_single_fold, export_evaluation, probability_metrics,
)
from src.models.ml1.indicator_validation import FeatureVariantData
from src.models.ml1.market_indicators import ENRICHED_COLUMNS, variant_profile
from src.models.ml1.time_validation import (
    FEATURE_COLUMNS, FINAL_TEST_START, LABEL, WalkForwardConfig, prepare_development_data,
)


def frame(n, *, vix=None, start="2010-01-04"):
    index = pd.bdate_range(start, periods=n, name="Date")
    columns = FEATURE_COLUMNS if vix is None else ENRICHED_COLUMNS
    X = pd.DataFrame(0., index=index, columns=columns)
    if vix is not None:
        X["VIX_level"] = vix
    return X


def labels(X, values):
    return pd.Series(values, index=X.index, name=LABEL)


def test_frequency_is_only_the_observed_training_rate_and_probabilities_sum_to_one():
    X = frame(4)
    y = labels(X,[0,1,0,0])
    model = HistoricalFrequencyBaseline().fit(X,y)
    p = model.predict_proba(frame(3,start="2011-01-03"))
    np.testing.assert_allclose(p,[[.75,.25]]*3)
    assert np.all(p>=0) and np.all(p<=1)
    np.testing.assert_allclose(p.sum(axis=1),1)
    np.testing.assert_array_equal(model.classes_,[0,1])
    assert model.training_summary_["n_positive_train"] == 1
    y.iloc[:] = 1
    assert model.probability_ == .25


@pytest.mark.parametrize("model",[HistoricalFrequencyBaseline(),VixThresholdBaseline()])
def test_predict_requires_a_train_fit(model):
    with pytest.raises(NotFittedError):
        model.predict_proba(frame(2))


@pytest.mark.parametrize("truth",[0,1])
def test_frequency_single_class_training_is_still_a_defined_frequency(truth):
    X = frame(4)
    model = HistoricalFrequencyBaseline().fit(X,labels(X,[truth]*4))
    assert model.predict_proba(frame(1,start="2011-01-03"))[0,1] == truth


def test_vix_equality_25_belongs_to_high_group_and_laplace_formula_is_exact():
    X = frame(4,vix=[20,24.999,25,30])
    model = VixThresholdBaseline(min_group_size=2).fit(X,labels(X,[0,0,1,1]))
    predictions = model.predict_proba(frame(3,vix=[24.999,25,30],start="2011-01-03"))[:,1]
    np.testing.assert_allclose(predictions,[.25,.75,.75])
    assert model.training_summary_["groups"]["high"]["n_train"] == 2
    assert model.get_params() == {"threshold":25.,"alpha":1.,"min_group_size":2}


def test_default_experiment_is_fixed_before_validation():
    assert VixThresholdBaseline().get_params() == {"threshold":25.,"alpha":1.,"min_group_size":30}
    with pytest.raises(ValueError,match="fixes the VIX threshold at 25"):
        VixThresholdBaseline(threshold=20)


def test_empty_group_falls_back_to_global_training_frequency():
    X = frame(60,vix=[20]*60)
    y = labels(X,[1]*6+[0]*54)
    model = VixThresholdBaseline().fit(X,y)
    assert model.training_summary_["groups"]["high"]["fallback_to_global"] is True
    p = model.predict_proba(frame(2,vix=[25,20],start="2011-01-03"))[:,1]
    assert p[0] == pytest.approx(.1)
    assert p[1] == pytest.approx(7/62)


@pytest.mark.parametrize("high_count,expected_fallback",[(29,True),(30,False)])
def test_small_group_fallback_has_an_explicit_support_boundary(high_count,expected_fallback):
    X = frame(60,vix=[30]*high_count+[20]*(60-high_count))
    y = labels(X,[1]*high_count+[0]*(60-high_count))
    model = VixThresholdBaseline().fit(X,y)
    group = model.training_summary_["groups"]["high"]
    assert group["fallback_to_global"] is expected_fallback
    expected = high_count/60 if expected_fallback else (high_count+1)/(high_count+2)
    assert group["probability"] == pytest.approx(expected)


def test_smoothing_avoids_group_extremes_without_assigning_arbitrary_zero_one():
    X = frame(60,vix=[20]*30+[30]*30)
    model = VixThresholdBaseline().fit(X,labels(X,[0]*30+[1]*30))
    p = model.predict_proba(frame(2,vix=[20,30],start="2011-01-03"))[:,1]
    np.testing.assert_allclose(p,[1/32,31/32])
    assert np.all((p>0)&(p<1))


@pytest.mark.parametrize("alpha",[0,-1,np.nan,np.inf])
def test_invalid_smoothing_is_rejected(alpha):
    with pytest.raises(ValueError,match="alpha"):
        VixThresholdBaseline(alpha=alpha)


def test_vix_baseline_uses_only_vix_level_for_its_bucket():
    X = frame(60,vix=[20]*30+[30]*30)
    model = VixThresholdBaseline().fit(X,labels(X,[0]*30+[1]*30))
    validation = frame(4,vix=[20,30,25,10],start="2011-01-03")
    original = model.predict_proba(validation)
    other = validation.copy()
    for col in other:
        if col != "VIX_level":
            other[col] = 99999
    np.testing.assert_array_equal(model.predict_proba(other),original)
    with pytest.raises(DataValidationError,match="feature schema"):
        model.predict_proba(validation.rename(columns={"VIX_level":"VIXCLS"}))


@pytest.mark.parametrize("kind",["frequency","vix"])
def test_predictions_are_invariant_when_only_y_validation_changes(kind):
    enriched = kind == "vix"
    X_train = frame(60,vix=[20]*30+[30]*30 if enriched else None)
    y_train = labels(X_train,[0]*30+[1]*30)
    X_validation = frame(8,vix=[20,30]*4 if enriched else None,start="2011-01-03")
    y_validation = labels(X_validation,[0,1]*4)
    factory = VixThresholdBaseline if enriched else HistoricalFrequencyBaseline
    p1,m1,_ = evaluate_single_fold(factory(),X_train,y_train,X_validation,y_validation)
    p2,m2,_ = evaluate_single_fold(factory(),X_train,y_train,X_validation,1-y_validation)
    np.testing.assert_array_equal(p1,p2)


@pytest.mark.parametrize("kind",["frequency","vix","metrics"])
def test_2023_label_values_are_rejected_before_inspection(kind):
    class ForbiddenLabel:
        def __eq__(self,other):
            raise AssertionError("A final-test label value was inspected")
    X = frame(3,vix=[20,30,25] if kind == "vix" else None,start="2023-01-03")
    y = labels(X,[ForbiddenLabel() for _ in range(3)])
    with pytest.raises(DataValidationError,match="2023"):
        if kind == "metrics":
            probability_metrics(y,[.1]*3)
        else:
            model = VixThresholdBaseline() if kind == "vix" else HistoricalFrequencyBaseline()
            model.fit(X,y)


@pytest.mark.parametrize("truth",[0,1])
def test_one_class_metrics_have_explicit_nan_ranking_metrics_and_defined_brier(truth):
    X = frame(4,start="2015-01-02")
    scores = probability_metrics(labels(X,[truth]*4),[.2]*4)
    assert np.isnan(scores["average_precision"]) and np.isnan(scores["roc_auc"])
    assert scores["brier_score"] == pytest.approx((truth-.2)**2)
    assert scores["metric_status"].startswith("single_class")


def test_average_precision_is_ap_and_brier_is_mean_squared_probability_error():
    X = frame(4,start="2015-01-02")
    y = labels(X,[0,1,0,1])
    scores = probability_metrics(y,[.1,.8,.2,.7])
    assert scores["average_precision"] == pytest.approx(1)
    assert scores["roc_auc"] == pytest.approx(1)
    assert scores["brier_score"] == pytest.approx(np.mean((y.to_numpy()-[.1,.8,.2,.7])**2))


def test_average_precision_uses_stepwise_recall_weights_not_trapezoidal_area():
    X = frame(4,start="2015-01-02")
    scores = probability_metrics(labels(X,[0,1,1,0]),[.9,.8,.7,.1])
    # Positive examples have ranks 2 and 3, with precision 1/2 and 2/3.
    assert scores["average_precision"] == pytest.approx((1/2+2/3)/2)


@pytest.mark.parametrize("probability",[[.1],[.1,.2,.3,np.nan],[.1,.2,.3,1.1],[.1,.2,.3,-.1]])
def test_invalid_metric_probabilities_are_rejected(probability):
    X = frame(4,start="2015-01-02")
    with pytest.raises(DataValidationError,match="probabilities"):
        probability_metrics(labels(X,[0,1,0,1]),probability)


@pytest.fixture
def paired_data():
    sessions = xcals.get_calendar("ARCX",start="2013-01-01",end="2016-03-31").sessions
    dates = sessions[(sessions >= "2013-01-03")&(sessions < "2016-01-01")]
    positions = sessions.get_indexer(dates)
    X = pd.DataFrame(0.,index=dates,columns=FEATURE_COLUMNS)
    y = pd.Series((np.arange(len(dates))%11 == 0).astype(int),index=dates,name=LABEL)
    controls = pd.DataFrame({"target_start_session":sessions[positions+1],
                             "target_end_session":sessions[positions+21],"future_session_count":21},index=dates)
    development = prepare_development_data(X,y,controls,sessions)
    enriched = development.X.copy()
    # Synthetic proxy derived from previous-session values, never contemporaneous ones.
    provider = pd.Series(np.where(np.arange(len(sessions))%20 < 10,20.,30.),index=sessions)
    enriched["VIX_level"] = provider.shift(1).reindex(enriched.index)
    enriched["VIX_change_5d"] = 0.
    core = FeatureVariantData(development,development.X.copy(),variant_profile(),pd.DatetimeIndex([]))
    vix = FeatureVariantData(development,enriched,{**variant_profile("core_vix",True),"lag_spy_sessions":1},pd.DatetimeIndex([]))
    config = WalkForwardConfig(validation_years=(2015,),min_train_observations=10,min_validation_observations=10)
    return core,vix,config


def test_evaluator_respects_folds_purge_dates_and_experimental_status(paired_data):
    core,vix,config = paired_data
    result = evaluate_baselines(core,vix,config)
    assert len(result.metrics) == 4
    assert set(result.predictions.baseline) == {"historical_frequency","vix_threshold_25"}
    for name,group in result.predictions.groupby("baseline"):
        assert group.prediction_date.is_monotonic_increasing
        assert not group.prediction_date.duplicated().any()
        assert group.prediction_date.dt.year.eq(2015).all()
        assert group.train_end.lt(group.prediction_date).all()
        assert group.train_target_end_max.lt(group.prediction_date.min()).all()
        assert group.experimental.eq(name=="vix_threshold_25").all()
        assert np.all((group.predicted_probability>=0)&(group.predicted_probability<=1))
    assert result.comparison_plan[0]["purged_core"] == result.comparison_plan[0]["purged_vix"] == 21


def test_whole_single_fold_evaluation_does_not_use_its_validation_to_fit(paired_data):
    core,vix,config = paired_data
    first = evaluate_baselines(core,vix,config)
    changed_y = core.core.y.copy()
    changed_y.loc[changed_y.index.year==2015] = 1-changed_y.loc[changed_y.index.year==2015]
    changed_core = replace(core,core=replace(core.core,y=changed_y))
    changed_vix = replace(vix,core=replace(vix.core,y=changed_y))
    second = evaluate_baselines(changed_core,changed_vix,config)
    np.testing.assert_array_equal(first.predictions.predicted_probability,second.predictions.predicted_probability)


@pytest.mark.parametrize("lag",[0,2])
def test_evaluator_requires_the_existing_one_session_vix_variant(paired_data,lag):
    core,vix,config = paired_data
    bad = replace(vix,profile={**vix.profile,"lag_spy_sessions":lag})
    with pytest.raises(DataValidationError,match="exactly-one-SPY-session"):
        evaluate_baselines(core,bad,config)


def test_common_dates_are_chosen_before_scores_and_differences_are_recorded(paired_data):
    core,vix,config = paired_data
    removed = vix.X.index[vix.X.index.year==2015][50]
    keep = vix.X.index != removed
    trimmed = replace(vix, X=vix.X.loc[keep],core=replace(vix.core,
        X=vix.core.X.loc[keep],y=vix.core.y.loc[keep],target_info=vix.core.target_info.loc[keep]))
    result = evaluate_baselines(core,trimmed,config)
    assert result.comparison_plan[0]["core_only_dates"] == [str(removed.date())]
    sets = [set(group.prediction_date) for _,group in result.predictions.groupby("baseline")]
    assert sets[0] == sets[1] and removed not in sets[0]


def test_evaluation_never_shuffles_or_reorders_randomly(paired_data,monkeypatch):
    import random
    def forbidden(*args,**kwargs):
        raise AssertionError("Shuffling is forbidden")
    monkeypatch.setattr(np.random,"shuffle",forbidden)
    monkeypatch.setattr(np.random,"permutation",forbidden)
    monkeypatch.setattr(random,"shuffle",forbidden)
    evaluate_baselines(*paired_data)


def test_exports_are_reproducible_and_never_overwrite(paired_data,tmp_path):
    first = evaluate_baselines(*paired_data)
    second = evaluate_baselines(*paired_data)
    pd.testing.assert_frame_equal(first.metrics,second.metrics)
    pd.testing.assert_frame_equal(first.predictions,second.predictions)
    one,two = tmp_path/"one",tmp_path/"two"
    export_evaluation(first,one)
    export_evaluation(second,two)
    assert set(p.name for p in one.iterdir()) == set(RESULT_FILES)
    for name in RESULT_FILES:
        assert (one/name).read_bytes() == (two/name).read_bytes(),name
    with pytest.raises(FileExistsError):
        export_evaluation(second,one)
    exported = pd.read_parquet(one/"baseline_predictions.parquet")
    assert exported.prediction_date.lt(FINAL_TEST_START).all()
    assert {"fold","validation_year","feature_variant","experimental","baseline_parameters","observed_target"}.issubset(exported)


def test_evaluation_cannot_export_2023_predictions(paired_data,tmp_path):
    result = evaluate_baselines(*paired_data)
    bad = result.predictions.copy()
    bad.loc[0,"prediction_date"] = pd.Timestamp("2023-01-03")
    with pytest.raises(DataValidationError,match="2023"):
        export_evaluation(replace(result,predictions=bad),tmp_path/"forbidden")


def test_real_evaluation_protects_final_label_reads_and_previous_modules(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    if not (root/"data/processed/ml1_core_vix_experimental/dataset_metadata.json").exists():
        pytest.skip("Local generated data not distributed in Git")
    protected = [root/"src/models/ml1"/name for name in
                 ["core.py","pipeline.py","time_validation.py","market_indicators.py","indicator_validation.py"]]
    protected += list((root/"data/processed/ml1").iterdir())
    protected += list((root/"data/processed/ml1_core_vix_experimental").iterdir())
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    reader = pd.read_parquet
    reads = []
    def guarded(path,*args,**kwargs):
        if Path(path).name == "targets_ml1.parquet":
            columns = kwargs.get("columns")
            assert columns is not None
            if LABEL in columns:
                assert columns == [LABEL]
                assert ("Date","<",FINAL_TEST_START) in kwargs.get("filters",[])
                assert ("target_end_session","<",FINAL_TEST_START) in kwargs.get("filters",[])
                reads.append(kwargs["filters"])
        return reader(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    result = evaluate_baselines()
    assert len(reads) == 2
    assert len(result.metrics) == 18
    assert result.predictions.prediction_date.max() == pd.Timestamp("2022-11-30")
    assert result.predictions.validation_year.nunique() == 8
    for name,group in result.predictions.groupby("baseline"):
        pooled = result.metrics.loc[(result.metrics.baseline==name)&(result.metrics.period_type=="pooled_oof")].iloc[0]
        assert pooled.n_observations == len(group)
        assert pooled.n_positive == group.observed_target.sum()
        assert pooled.brier_score == pytest.approx(np.mean((group.observed_target-group.predicted_probability)**2))
    assert before == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
