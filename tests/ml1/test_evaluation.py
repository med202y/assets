"""Evaluate recorded probabilities only: these tests never fit an estimator."""

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.ml1 import evaluate_ml1 as evaluation
from src.models.ml1.core import DataValidationError
from src.models.ml1.time_validation import FINAL_TEST_START, LABEL, build_walk_forward_folds, load_development_data


def labels(values, start="2015-01-02"):
    return pd.Series(values, index=pd.bdate_range(start,periods=len(values)),name=LABEL)


def test_synthetic_ap_is_step_weighted_not_trapezoidal_and_brier_is_mse():
    y=labels([0,1,1,0]);p=np.array([.9,.8,.7,.1])
    m=evaluation.probability_metrics(y,p)
    assert m["average_precision"]==pytest.approx((1/2+2/3)/2)
    assert m["brier_score"]==pytest.approx(np.mean((y.to_numpy()-p)**2))
    assert m["roc_auc"]==pytest.approx(.5)
    assert m["n_positive"]==m["n_negative"]==2


def test_perfect_probabilities_have_ap_auc_one_and_brier_zero():
    m=evaluation.probability_metrics(labels([0,1,0,1]),[0,1,0,1])
    assert m["average_precision"]==m["roc_auc"]==1
    assert m["brier_score"]==0


@pytest.mark.parametrize("truth",[0,1])
def test_single_class_has_justified_nan_ranking_metrics_and_defined_brier(truth):
    m=evaluation.probability_metrics(labels([truth]*4),[.2]*4)
    assert np.isnan(m["average_precision"]) and np.isnan(m["roc_auc"])
    assert m["brier_score"]==pytest.approx((truth-.2)**2)
    assert m["ranking_metric_reason"] and m["metric_status"].startswith("single_class")


@pytest.mark.parametrize("p",[[.1],[-.1,.2,.3,.4],[.1,.2,np.nan,.4],[.1,.2,np.inf,.4],[.1,.2,1.1,.4]])
def test_invalid_metric_probabilities_are_rejected(p):
    with pytest.raises(DataValidationError,match="Probabilities"):
        evaluation.probability_metrics(labels([0,1,0,1]),p)


def test_final_labels_are_rejected_before_value_inspection():
    class Forbidden:
        def __eq__(self,other):raise AssertionError("Final label inspected")
    y=labels([Forbidden(),Forbidden()],start="2023-01-03")
    with pytest.raises(DataValidationError,match="2023"):
        evaluation.probability_metrics(y,[.1,.2])


@pytest.fixture(scope="module")
def inputs():
    if not (evaluation.DEFAULT_TRAINING/"training_metadata.json").exists():
        pytest.skip("Local OOF predictions not distributed in Git")
    data=load_development_data()
    folds=build_walk_forward_folds(data)
    ml=pd.read_parquet(evaluation.DEFAULT_TRAINING/"ml_models_oof_predictions.parquet")
    baseline=pd.read_parquet(evaluation.DEFAULT_BASELINES/"baseline_predictions.parquet",
                             filters=[("prediction_date","<",FINAL_TEST_START)])
    metadata=json.loads((evaluation.DEFAULT_TRAINING/"training_metadata.json").read_text())
    return ml,baseline,data,folds,metadata


@pytest.fixture(scope="module")
def result():
    if not (evaluation.DEFAULT_TRAINING/"training_metadata.json").exists():
        pytest.skip("Local OOF predictions not distributed in Git")
    return evaluation.evaluate_saved_predictions()


def test_loader_uses_only_filtered_targets_and_saved_predictions(monkeypatch):
    if not (evaluation.DEFAULT_TRAINING/"training_metadata.json").exists():pytest.skip("Local data absent")
    original=pd.read_parquet;reads=[]
    def guarded(path,*args,**kwargs):
        if Path(path).name=="targets_ml1.parquet":
            assert kwargs.get("columns") is not None
            if LABEL in kwargs["columns"]:
                assert kwargs["columns"]==[LABEL]
                assert ("Date","<",FINAL_TEST_START) in kwargs.get("filters",[])
                assert ("target_end_session","<",FINAL_TEST_START) in kwargs.get("filters",[])
                reads.append(path)
        if Path(path).name=="baseline_predictions.parquet" and "observed_target" in kwargs.get("columns",[]):
            assert ("prediction_date","<",FINAL_TEST_START) in kwargs.get("filters",[])
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    r=evaluation.evaluate_saved_predictions()
    assert len(reads)==1 and len(r.predictions)==5*1993
    assert r.provenance["new_fits"]==0


def test_alignment_keeps_all_dates_and_identical_true_labels(inputs):
    ml,b,data,folds,_=inputs
    joined,diagnostic=evaluation.align_predictions(ml,b,data,folds)
    assert diagnostic["expected_dates"]==1993
    for name,g in joined.groupby("approach"):
        assert len(g)==1993
        assert pd.DatetimeIndex(g.prediction_date).is_monotonic_increasing
        np.testing.assert_array_equal(g.observed_target,data.y.loc[pd.DatetimeIndex(g.prediction_date)])
        assert g.experimental.eq(name=="vix_threshold_25").all()


@pytest.mark.parametrize("mutation",["duplicate","nan","inf","negative","above_one","2023","wrong_year","unknown_model"])
def test_corrupt_predictions_are_rejected_before_scoring(mutation,inputs):
    ml,b,data,folds,_=inputs;ml=ml.copy()
    if mutation=="duplicate":ml=pd.concat([ml,ml.iloc[:1]],ignore_index=True)
    elif mutation=="2023":ml.loc[0,"prediction_date"]=pd.Timestamp("2023-01-03")
    elif mutation=="wrong_year":ml.loc[0,"validation_year"]=2021
    elif mutation=="unknown_model":ml.loc[0,"model"]="tuned_model"
    else:ml.loc[0,"predicted_probability"]={"nan":np.nan,"inf":np.inf,"negative":-.1,"above_one":1.1}[mutation]
    with pytest.raises(DataValidationError):evaluation.align_predictions(ml,b,data,folds)


def test_missing_dates_yield_exact_diagnostic_without_intersection_or_scores(inputs):
    ml,b,data,folds,_=inputs
    removed=ml.iloc[0]
    with pytest.raises(evaluation.AlignmentError) as caught:
        evaluation.align_predictions(ml.iloc[1:],b,data,folds)
    row=caught.value.diagnostic["approaches"][removed.model]
    assert row["missing_dates"]==[str(removed.prediction_date.date())]


def test_baseline_label_disagreement_blocks_comparison(inputs):
    ml,b,data,folds,_=inputs;b=b.copy()
    b.loc[0,"observed_target"]=1-b.loc[0,"observed_target"]
    with pytest.raises(DataValidationError,match="baseline labels differ"):
        evaluation.align_predictions(ml,b,data,folds)


@pytest.mark.parametrize("column,value",[("experimental",False),("feature_variant","core"),("vix_lag_spy_sessions",0)])
def test_vix_cannot_lose_experimental_status_or_lag(column,value,inputs):
    ml,b,data,folds,_=inputs;b=b.copy()
    b.loc[b.baseline=="vix_threshold_25",column]=value
    with pytest.raises(DataValidationError,match="EXPERIMENTAL"):
        evaluation.align_predictions(ml,b,data,folds)


@pytest.mark.parametrize("mutation",["schema","configuration","data_fingerprint","fold_provenance","temporal_column"])
def test_provenance_changes_are_rejected(mutation,inputs):
    ml,_,data,folds,metadata=inputs
    ml=ml.copy();metadata=json.loads(json.dumps(metadata))
    if mutation=="schema":ml["future_min_price_21"]=1.
    elif mutation=="configuration":ml.loc[0,"configuration_id"]="different"
    elif mutation=="data_fingerprint":metadata["dataframe_sha256"]["y_pre2023"]="0"*64
    elif mutation=="fold_provenance":metadata["folds"][0]["n_purged"]=0
    else:ml.loc[0,"train_target_end_max"]=pd.Timestamp("2015-01-02")
    with pytest.raises(DataValidationError):evaluation.validate_training_provenance(ml,metadata,data,folds)


def test_metric_rows_counts_pooling_and_na_policy(result):
    assert len(result.metrics)==45
    for name,g in result.predictions.groupby("approach"):
        pooled=result.metrics.loc[(result.metrics.approach==name)&(result.metrics.period_type=="pooled_oof")].iloc[0]
        assert pooled.n_observations==1993 and pooled.n_positive==206 and pooled.n_negative==1787
        assert pooled.brier_score==pytest.approx(np.mean((g.observed_target-g.predicted_probability)**2))
    na=result.metrics.loc[result.metrics.validation_year.isin([2017,2019])]
    assert len(na)==10 and na.average_precision.isna().all() and na.roc_auc.isna().all()
    assert na.brier_score.notna().all() and na.ranking_metric_reason.str.len().gt(0).all()


def test_baselines_match_original_csv_and_rounded_historical_values(result):
    assert result.baseline_verification.historical_csv_match.all()
    assert (result.baseline_verification[list(evaluation.METRIC_NAMES)]<=1e-6).all().all()


def test_baseline_metric_mismatch_blocks_recommendation(result):
    saved=pd.read_csv(evaluation.DEFAULT_BASELINES/"baseline_metrics.csv")
    saved.loc[0,"average_precision"]+=.01
    with pytest.raises(DataValidationError,match="history mismatch"):
        evaluation.verify_baseline_metrics(result.metrics,saved)


def test_selection_is_reproducible_and_only_provisional(result):
    first=evaluation.provisional_recommendation(result.metrics)
    second=evaluation.provisional_recommendation(result.metrics.sample(frac=1,random_state=42))
    assert first==second
    assert first["candidate"]=="logistic_regression"
    assert first["status"]=="CANDIDAT PROVISOIRE"
    assert first["statistical_superiority_established"] is False


@pytest.mark.parametrize("reason",["tie","below_baselines"])
def test_no_candidate_is_forced_when_ap_rule_cannot_choose(reason,result):
    metrics=result.metrics.copy()
    if reason=="tie":metrics.loc[metrics.approach.isin(evaluation.MODELS)&metrics.period_type.eq("pooled_oof"),"average_precision"]=.2
    else:metrics.loc[metrics.approach.isin(evaluation.MODELS)&metrics.period_type.eq("pooled_oof"),"average_precision"]=.001
    assert evaluation.provisional_recommendation(metrics)["candidate"] is None


def test_exports_and_figures_are_created_without_overwriting(result,tmp_path):
    folder=tmp_path/"evaluation"
    evaluation.export_evaluation(result,folder)
    assert {p.name for p in folder.iterdir()}==set(evaluation.OUTPUT_FILES)
    assert len(pd.read_csv(folder/"model_metrics.csv"))==45
    for p in folder.glob("*.png"):
        assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and p.stat().st_size>10000
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    with pytest.raises(FileExistsError):evaluation.export_evaluation(result,folder)
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}


def test_unverified_baselines_cannot_be_exported(result,tmp_path):
    bad=replace(result,provenance={**result.provenance,"baseline_metrics_verified":False})
    with pytest.raises(DataValidationError,match="Unverified baselines"):
        evaluation.export_evaluation(bad,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_evaluation_preserves_existing_sources_reports_and_data():
    protected=list((evaluation.PROJECT_ROOT/"src/models/ml1").glob("*.py"))
    for directory in ["reports/ml1/baselines","data/processed/ml1","data/processed/ml1_training"]:
        path=evaluation.PROJECT_ROOT/directory
        if not path.exists():pytest.skip("Local data absent")
        protected.extend(p for p in path.iterdir() if p.is_file())
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    evaluation.evaluate_saved_predictions()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}


def test_evaluation_module_cannot_construct_fit_tune_or_calibrate_estimators():
    tree=ast.parse(Path(evaluation.__file__).read_text())
    forbidden={"fit","fit_transform","train_walk_forward","fit_predict_fold","build_estimator","xgboost_preflight",
               "GridSearchCV","RandomizedSearchCV","CalibratedClassifierCV","LogisticRegression","RandomForestClassifier","XGBClassifier"}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
            assert name not in forbidden
        if isinstance(node,ast.ImportFrom):
            assert node.module not in {"train_ml1","evaluate_baselines","sklearn.model_selection","sklearn.calibration"}
