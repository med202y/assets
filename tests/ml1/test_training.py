"""Fixed supervised training tests; no new-model performance scoring."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.models.ml1.core import DataValidationError
from src.models.ml1.time_validation import (
    FEATURE_COLUMNS, FINAL_TEST_START, LABEL, WalkForwardConfig,
    build_walk_forward_folds, load_development_data, prepare_development_data,
)
from src.models.ml1 import train_ml1 as training


@pytest.fixture(scope="module")
def data():
    sessions = xcals.get_calendar("ARCX", start="2013-01-01", end="2016-03-31").sessions
    dates = sessions[(sessions >= "2013-01-03") & (sessions < "2016-01-01")]
    positions = sessions.get_indexer(dates)
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(len(dates),10)),
                     index=dates, columns=FEATURE_COLUMNS)
    X.loc[X.index.year==2015] += 5  # Make validation distribution visibly distinct.
    y = pd.Series((np.arange(len(dates))%11==0).astype(int), index=dates, name=LABEL)
    info = pd.DataFrame({"target_start_session":sessions[positions+1],
                         "target_end_session":sessions[positions+21],
                         "future_session_count":21}, index=dates)
    return prepare_development_data(X,y,info,sessions)


@pytest.fixture(scope="module")
def config():
    return WalkForwardConfig(validation_years=(2015,),min_train_observations=10,
                             min_validation_observations=10)


@pytest.fixture(scope="module")
def fold(data,config):
    return build_walk_forward_folds(data,config)[0]


@pytest.fixture(scope="module")
def native_check():
    return training.xgboost_preflight()


@pytest.mark.parametrize("name",training.MODEL_NAMES)
def test_estimators_are_cloneable_sklearn_classifiers_with_exact_fixed_parameters(name,native_check):
    if name=="xgboost" and native_check["status"]!="passed":
        pytest.skip(f"XGBoost unavailable: {native_check}")
    model=training.build_estimator(name)
    assert is_classifier(model)
    cloned=clone(model)
    classifier=cloned.named_steps["classifier"] if isinstance(cloned,Pipeline) else cloned
    for parameter,value in training.configuration_spec(name)["parameters"].items():
        assert classifier.get_params()[parameter]==value
    if name=="logistic_regression":
        assert list(cloned.named_steps)==["scaler","classifier"]
    else:
        assert not isinstance(cloned,Pipeline)


@pytest.mark.parametrize("name",training.MODEL_NAMES)
def test_fit_proba_shape_reproducibility_and_y_validation_invariance(name,data,fold,native_check):
    if name=="xgboost" and native_check["status"]!="passed":
        pytest.skip(f"XGBoost unavailable: {native_check}")
    first=training.fit_predict_fold(name,data,fold)
    changed_y=data.y.copy()
    changed_y.iloc[fold.validation_indices]=1-changed_y.iloc[fold.validation_indices]
    # A second full fit proves both reproducibility and independence from y_validation.
    second=training.fit_predict_fold(name,replace(data,y=changed_y),fold)
    p=first.probabilities
    assert p.shape==(len(fold.validation_indices),)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.testing.assert_array_equal(p,second.probabilities)
    np.testing.assert_array_equal(first.estimator.classes_,[0,1])
    if name=="logistic_regression":
        scaler=first.estimator.named_steps["scaler"]
        train=data.X.iloc[fold.train_indices]
        np.testing.assert_allclose(scaler.mean_,train.mean(axis=0))
        np.testing.assert_allclose(scaler.var_,train.var(axis=0,ddof=0))
        assert scaler.n_samples_seen_==len(train)
        assert not np.allclose(scaler.mean_,data.X.mean(axis=0))


def test_fit_receives_only_train_and_never_eval_set(data,fold,monkeypatch):
    calls=[]
    class Spy:
        classes_=np.array([0,1])
        def fit(self,X,y,**kwargs):
            assert X.index.equals(data.X.index[fold.train_indices])
            np.testing.assert_array_equal(y,data.y.iloc[fold.train_indices])
            assert not kwargs
            calls.append("fit")
        def predict_proba(self,X):
            assert calls==["fit"]
            assert X.index.equals(data.X.index[fold.validation_indices])
            calls.append("predict")
            return np.tile([.8,.2],(len(X),1))
    monkeypatch.setattr(training,"build_estimator",lambda name:Spy())
    training.fit_predict_fold("logistic_regression",data,fold)
    assert calls==["fit","predict"]


@pytest.mark.parametrize("where",["train","validation"])
@pytest.mark.parametrize("value",[np.nan,np.inf,-np.inf])
def test_nonfinite_features_rejected_before_fit(where,value,data,fold,monkeypatch):
    X=data.X.copy()
    indices=fold.train_indices if where=="train" else fold.validation_indices
    X.iloc[indices[0],0]=value
    monkeypatch.setattr(training,"build_estimator",lambda name:pytest.fail("fit must not start"))
    with pytest.raises(DataValidationError,match="finite"):
        training.fit_predict_fold("logistic_regression",replace(data,X=X),fold)


@pytest.mark.parametrize("label",[0,1,np.nan,2])
def test_single_class_missing_and_invalid_training_labels_rejected(label,data,fold,monkeypatch):
    y=data.y.astype(float).copy()
    y.iloc[fold.train_indices]=label
    monkeypatch.setattr(training,"build_estimator",lambda name:pytest.fail("fit must not start"))
    with pytest.raises(DataValidationError,match="both classes|complete and binary"):
        training.fit_predict_fold("random_forest",replace(data,y=y),fold)


@pytest.mark.parametrize("bad_column",["future_min_price_21","target_end_session","VIX_level","y_crisis_21d"])
def test_future_metadata_labels_and_vix_cannot_enter_features(bad_column,data,fold):
    X=data.X.copy();X[bad_column]=1.
    with pytest.raises(DataValidationError,match="separate features"):
        training.fit_predict_fold("random_forest",replace(data,X=X),fold)


def test_reordering_core_features_is_rejected(data,fold):
    with pytest.raises(DataValidationError):
        training.fit_predict_fold("random_forest",replace(data,X=data.X.iloc[:,::-1]),fold)


def test_unpurged_target_is_rejected_before_fit(data,fold,monkeypatch):
    indices=np.sort(np.append(fold.train_indices,fold.purged_train_indices[0]))
    monkeypatch.setattr(training,"build_estimator",lambda name:pytest.fail("fit must not start"))
    with pytest.raises(DataValidationError,match="reaches validation"):
        training.fit_predict_fold("random_forest",data,replace(fold,train_indices=indices))


def test_overlapping_train_validation_is_rejected(data,fold):
    indices=np.append(fold.train_indices,fold.validation_indices[0])
    with pytest.raises(DataValidationError,match="no common date"):
        training.fit_predict_fold("random_forest",data,replace(fold,train_indices=indices))


def test_2023_guard_runs_before_label_value_inspection():
    class Forbidden:
        def __eq__(self,other):
            raise AssertionError("Final label inspected")
    dates=pd.bdate_range("2023-01-03",periods=4)
    X=pd.DataFrame(0.,index=dates,columns=FEATURE_COLUMNS)
    y=pd.Series([Forbidden() for _ in dates],index=dates,name=LABEL)
    with pytest.raises(DataValidationError,match="2023"):
        training.validate_training_inputs(X,y,X)


@pytest.mark.parametrize("matrix",[
    [[.8,.2]], [[.8,np.nan],[.8,.2]], [[.8,1.2],[.8,.2]],
    [[.8,.3],[.8,.2]], [[.8,-.2],[.8,.2]],
])
def test_invalid_native_probability_outputs_rejected(matrix,data):
    class Bad:
        classes_=np.array([0,1])
        def predict_proba(self,X):return matrix
    with pytest.raises(DataValidationError,match="predict_proba"):
        training.positive_probabilities(Bad(),data.X.iloc[:2])


def test_class_one_is_extracted_by_class_name_not_position(data):
    class Reversed:
        classes_=np.array([1,0])
        def predict_proba(self,X):return np.tile([.3,.7],(len(X),1))
    np.testing.assert_allclose(training.positive_probabilities(Reversed(),data.X.iloc[:3]),.3)


def test_native_preflight_records_libraries_and_probability_shape(native_check):
    assert native_check["status"]=="passed",native_check
    assert native_check["probability_shape"]==[4,2]
    if native_check["machine"]=="arm64":
        assert all(p["exists"] for p in native_check["libraries"])
        assert any(p["internal_api"]=="openmp" for p in native_check["threadpools"])


@pytest.mark.parametrize("returncode",[1,-6,-11])
def test_native_failure_does_not_crash_parent_and_is_documented(returncode,monkeypatch):
    def broken(*args,**kwargs):
        return subprocess.CompletedProcess(args[0],returncode,"","native loader failure")
    monkeypatch.setattr(training.subprocess,"run",broken)
    evidence=training.xgboost_preflight()
    assert evidence["status"]=="failed"
    assert evidence["returncode"]==returncode and "native loader" in evidence["stderr"]


def test_missing_xgboost_preserves_other_models_without_full_success(data,config,tmp_path,monkeypatch):
    monkeypatch.setattr(training,"xgboost_preflight",lambda:{"status":"failed","error":"ImportError: missing xgboost"})
    original=training.build_estimator
    def guarded(name):
        assert name!="xgboost"
        return original(name)
    monkeypatch.setattr(training,"build_estimator",guarded)
    result=training.train_walk_forward(data,config)
    assert result.metadata["status"]=="partial"
    assert result.metadata["successful_fits"]==2
    assert result.metadata["attempted_fits"]==2
    assert set(result.predictions.model)=={"logistic_regression","random_forest"}
    assert result.metadata["records"][-1]["status"]=="unavailable"
    training.export_training(result,tmp_path/"partial")
    assert (tmp_path/"partial/training_metadata.json").is_file()


def test_fit_error_preserves_successful_models_and_does_not_retune(data,config,monkeypatch):
    monkeypatch.setattr(training,"xgboost_preflight",lambda:{"status":"passed"})
    original=training.build_estimator
    calls=[]
    def guarded(name):
        calls.append(name)
        if name=="xgboost":raise ImportError("simulated late loader error")
        return original(name)
    monkeypatch.setattr(training,"build_estimator",guarded)
    result=training.train_walk_forward(data,config)
    assert result.metadata["status"]=="partial" and result.metadata["successful_fits"]==2
    assert result.metadata["records"][-1]["error_type"]=="ImportError"
    assert calls==list(training.MODEL_NAMES)


def test_convergence_warning_is_a_failure_with_no_silent_retuning(data,fold,monkeypatch):
    import warnings
    class Nonconverging:
        classes_=np.array([0,1])
        def fit(self,X,y):warnings.warn("fixed configuration did not converge",ConvergenceWarning)
        def predict_proba(self,X):return np.tile([.8,.2],(len(X),1))
    monkeypatch.setattr(training,"build_estimator",lambda name:Nonconverging())
    with pytest.raises(RuntimeError,match="did not converge"):
        training.fit_predict_fold("logistic_regression",data,fold)


@pytest.fixture(scope="module")
def run(data,config):
    return training.train_walk_forward(data,config)


def test_oof_structure_metadata_and_no_performance_metrics(run,fold):
    assert run.metadata["successful_fits"]==3 and run.metadata["status"]=="complete"
    assert len(run.predictions)==3*len(fold.validation_indices)
    assert tuple(run.predictions.columns)==training.PREDICTION_COLUMNS
    assert set(run.predictions.model)==set(training.MODEL_NAMES)
    assert run.predictions.feature_variant.eq("core").all()
    assert run.metadata["feature_columns"]==list(FEATURE_COLUMNS)
    assert run.metadata["folds"][0]["n_purged"]==21
    assert run.metadata["policy"]["performance_metrics_computed"] is False
    assert not {"average_precision","brier_score","roc_auc","observed_target"}.intersection(run.predictions)
    assert all(not record["warnings"] for record in run.metadata["records"])
    assert run.metadata["resolved_estimator_parameters"]["xgboost"]["missing"]=={"nonfinite_float":"NaN"}
    json.dumps(run.metadata,allow_nan=False)


def test_export_has_two_files_and_refuses_overwrite_without_mutation(run,tmp_path):
    out=tmp_path/"training"
    training.export_training(run,out)
    assert set(p.name for p in out.iterdir())==set(training.TRAINING_FILES)
    loaded=pd.read_parquet(out/training.TRAINING_FILES[0])
    pd.testing.assert_frame_equal(loaded,run.predictions)
    metadata=json.loads((out/training.TRAINING_FILES[1]).read_text())
    assert metadata["prediction_sha256"]==hashlib.sha256((out/training.TRAINING_FILES[0]).read_bytes()).hexdigest()
    before={p:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):training.export_training(run,out)
    assert before=={p:p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize("mutation",["2023","nan","duplicate","future_column","missing_date"])
def test_invalid_export_is_rejected_without_creating_directory(mutation,run,tmp_path):
    p=run.predictions.copy()
    if mutation=="2023":p.loc[0,"prediction_date"]=pd.Timestamp("2023-01-03")
    elif mutation=="nan":p.loc[0,"predicted_probability"]=np.nan
    elif mutation=="duplicate":p=pd.concat([p,p.iloc[:1]],ignore_index=True)
    elif mutation=="future_column":p["future_min_price_21"]=1.
    else:p=p.iloc[1:]
    with pytest.raises(DataValidationError):training.export_training(replace(run,predictions=p),tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_all_real_folds_and_label_reads_remain_protected_without_real_fits(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    if not (root/"data/processed/ml1/dataset_metadata.json").exists():
        pytest.skip("Local generated data not distributed in Git")
    protected=[root/p for p in subprocess.check_output(["git","ls-files","-z"],cwd=root).decode().split('\0') if p]
    for directory in ("data/processed/ml1","data/processed/ml1_core_vix_experimental"):
        protected.extend(p for p in (root/directory).iterdir() if p.is_file())
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    reader=pd.read_parquet
    reads=[]
    def guarded(path,*args,**kwargs):
        if Path(path).name=="targets_ml1.parquet":
            columns=kwargs.get("columns")
            assert columns is not None
            if LABEL in columns:
                assert columns==[LABEL]
                assert ("Date","<",FINAL_TEST_START) in kwargs.get("filters",[])
                assert ("target_end_session","<",FINAL_TEST_START) in kwargs.get("filters",[])
                reads.append(kwargs["filters"])
        return reader(path,*args,**kwargs)
    class NoScoringSpy:
        classes_=np.array([0,1])
        def fit(self,X,y):
            assert X.index.max()<FINAL_TEST_START
            assert tuple(X.columns)==FEATURE_COLUMNS
        def predict_proba(self,X):return np.tile([.9,.1],(len(X),1))
        def get_params(self,deep=True):return {}
    monkeypatch.setattr(pd,"read_parquet",guarded)
    monkeypatch.setattr(training,"build_estimator",lambda name:NoScoringSpy())
    monkeypatch.setattr(training,"xgboost_preflight",lambda:{"status":"passed"})
    result=training.train_walk_forward()
    assert len(reads)==1 and result.metadata["successful_fits"]==24
    assert len(result.predictions)==5979
    assert result.predictions.prediction_date.max()==pd.Timestamp("2022-11-30")
    assert all(f["n_purged"]==21 for f in result.metadata["folds"])
    for _,group in result.predictions.groupby("model"):
        assert len(group)==1993 and not group.prediction_date.duplicated().any()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}


def test_training_module_has_no_scoring_selection_or_baseline_evaluation_imports():
    import ast
    tree=ast.parse(Path(training.__file__).read_text())
    imported=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and ("metrics" in name or "model_selection" in name or "evaluate_baselines" in name) for name in imported)


def test_unknown_model_is_rejected_before_any_fit():
    with pytest.raises(ValueError,match="Unknown fixed model"):
        training.build_estimator("optimized_model")
