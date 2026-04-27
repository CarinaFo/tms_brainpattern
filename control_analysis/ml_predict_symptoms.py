# ML stuff
# core libraries
import pandas as pd
import numpy as np
from pathlib import Path

#TODO test delta FO prediction models

# run in base python (3.12)

system='windows'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:")
else:
    "No available system path defined *windows* or *linux*"

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')


def load_and_prep_data(hmm_dir: Path, n_states: int, exclude_repeater: bool = False) -> pd.DataFrame:
    """
    Load and preprocess HMM demo questionnaire data.
    - shifts state labels to start at 1
    - fills demographics per patient
    - casts common columns to categorical
    """
    csv_path = hmm_dir / f"hmm_demo_hads_{n_states}.csv"
    df = pd.read_csv(csv_path)

    if exclude_repeater and "patient" in df.columns:
        df = df[~df["patient"].astype(str).str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients (n_states={n_states})")

    # states start at 1
    if "state" in df.columns:
        df["state"] = df["state"] + 1

    # fill demographics (first value per patient)
    for col in ["age", "gender", "responder", "group", "years_with_depression"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    # categorical conversion (safe)
    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df


def build_ml_dataset(hmm_dir: Path, n_states: int) -> pd.DataFrame:
    """
    Build one-row-per-patient ML dataset using:
    - FO at session 1 pre-TMS
    - demographics
    - baseline HADS scores/items
    Outcomes:
    - sym_s3   = HADS-D at session 3
    - sym_last = last available HADS-D after final EEG
    """
    df = load_and_prep_data(hmm_dir, n_states)

    # --- session 1, pre-TMS only ---
    base = df[
        (df["session"].astype(int) == 2) &
        (df["tms"].astype(str) == "post")
    ].copy()

    # --- FO features ---
    fo_wide = (
        base.pivot(index="patient", columns="state", values="fo")
        .add_prefix("fo_state")
        .reset_index()
    )

    # flatten column names if needed
    fo_wide.columns = [str(c) for c in fo_wide.columns]

    # --- demographics ---
    demo_cols = [c for c in ["patient", "age", "gender"] if c in base.columns]
    demo = base[demo_cols].drop_duplicates(subset=["patient"])

    # --- baseline symptoms ---
    # include HADS totals and items if present
    hads_cols = [c for c in base.columns if "hads" in c.lower() and c != "matched_hads_date"]
    symptoms = base[["patient"] + hads_cols].drop_duplicates(subset=["patient"])

    # --- session 3 outcome ---
    s3 = (
        df[
            (df["session"].astype(int) == 3) &
            (df["tms"].astype(str) == "pre")
        ][["patient", "hads_dep_total"]]
        .drop_duplicates(subset=["patient"])
        .rename(columns={"hads_dep_total": "sym_s3"})
    )

    # --- follow-up outcome: last available HADS after final EEG ---
    future_path = Path(f"{hmm_dir}/future_hads_after_final_eeg_{n_states}.csv")
    future_df = pd.read_csv(future_path)

    # keep only patients present in the filtered main dataframe
    valid_patients = set(df["patient"].astype(str).unique())
    future_df = future_df[future_df["patient"].astype(str).isin(valid_patients)].copy()

    # identify future depression column
    if "future_depression" in future_df.columns:
        future_symptom_col = "future_depression"
    elif "matched_depression" in future_df.columns:
        future_symptom_col = "matched_depression"
    else:
        raise ValueError("Could not find future depression column in future HADS file.")

    # identify future date column
    if "future_hads_date" in future_df.columns:
        future_date_col = "future_hads_date"
    elif "matched_hads_date" in future_df.columns:
        future_date_col = "matched_hads_date"
    else:
        raise ValueError("Could not find future HADS date column in future HADS file.")

    future_df[future_date_col] = pd.to_datetime(future_df[future_date_col], errors="coerce")
    future_df = future_df.dropna(subset=[future_date_col]).copy()

    # keep last available HADS per patient
    eot = (
        future_df.sort_values(["patient", future_date_col])
        .groupby("patient", as_index=False)
        .tail(1)
        .copy()
        .rename(columns={
            future_symptom_col: "sym_last",
            future_date_col: "last_hads_date"
        })
    )

    # --- merge ---
    Xy = (
        fo_wide
        .merge(demo, on="patient", how="inner")
        .merge(symptoms, on="patient", how="inner")
        .merge(s3, on="patient", how="left")
        .merge(eot[["patient", "sym_last", "last_hads_date"]], on="patient", how="left")
    )

    return Xy

# =========================
# ML implementation
# =========================
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

SEED = 42


def get_ml_features_and_target(Xy: pd.DataFrame, target: str):
    """
    Prepare X and y for ML.
    target: 'sym_s3' or 'sym_last'
    """
    if target not in Xy.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    # keep only rows with the chosen outcome
    d = Xy.dropna(subset=[target]).copy()

    # -----------------------------
    # Select predictors explicitly
    # -----------------------------
    fo_cols = [c for c in Xy.columns if c.startswith("fo_")]

    demo_cols = [c for c in ["age", "gender"] if c in Xy.columns]

    predictor_cols = fo_cols + demo_cols

    # keep only relevant columns
    d = Xy[predictor_cols + [target]].dropna(subset=[target]).copy()

    X = d[predictor_cols]
    y = d[target].astype(float)

    return X, y, d


def make_preprocessor(X: pd.DataFrame):
    """
    Numeric: median impute + scale
    Categorical: most-frequent impute + one-hot encode
    """
    categorical_cols = [c for c in X.columns if str(X[c].dtype) == "category" or X[c].dtype == object]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def get_model_grids():
    """
    Reasonable tuning grids for modest sample sizes.
    """
    return {
        "ridge": {
            "model": Ridge(random_state=SEED),
            "grid": {
                "model__alpha": np.logspace(-3, 3, 20),
            },
        },
        "elasticnet": {
            "model": ElasticNet(max_iter=10000, random_state=SEED),
            "grid": {
                "model__alpha": np.logspace(-3, 1, 15),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        "random_forest": {
            "model": RandomForestRegressor(random_state=SEED),
            "grid": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [2, 3, 4, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, None],
            },
        },
        "hist_gbm": {
            "model": HistGradientBoostingRegressor(random_state=SEED),
            "grid": {
                "model__max_depth": [2, 3, 4, None],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__max_iter": [100, 200, 400],
                "model__min_samples_leaf": [5, 10, 20],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        },
    }


def evaluate_models_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits_outer: int = 5,
    n_splits_inner: int = 4,
    scoring=None,
):
    """
    Nested CV to estimate generalization performance honestly.
    Returns:
      results_df: summary of CV performance
      fitted_searches: final tuned models fit on the full dataset
    """
    if scoring is None:
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(X)
    model_grids = get_model_grids()

    outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=SEED)
    inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=SEED)

    rows = []
    fitted_searches = {}

    for model_name, spec in model_grids.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", spec["model"]),
        ])

        search = GridSearchCV(
            estimator=pipe,
            param_grid=spec["grid"],
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        cv_res = cross_validate(
            search,
            X,
            y,
            cv=outer_cv,
            scoring=scoring,
            return_estimator=True,
            n_jobs=1,   # safer because GridSearch already parallelizes
        )

        rows.append({
            "model": model_name,
            "n": len(y),
            "r2_mean": np.mean(cv_res["test_r2"]),
            "r2_sd": np.std(cv_res["test_r2"]),
            "mae_mean": -np.mean(cv_res["test_mae"]),
            "mae_sd": np.std(-cv_res["test_mae"]),
            "rmse_mean": -np.mean(cv_res["test_rmse"]),
            "rmse_sd": np.std(-cv_res["test_rmse"]),
        })

        # fit one final tuned model on the full dataset
        search.fit(X, y)
        fitted_searches[model_name] = search

    results_df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    return results_df, fitted_searches


def get_feature_names_from_pipeline(search, X: pd.DataFrame):
    """
    Extract transformed feature names from the fitted pipeline.
    """
    pipe = search.best_estimator_
    pre = pipe.named_steps["preprocess"]

    try:
        feature_names = pre.get_feature_names_out()
        return feature_names
    except Exception:
        return np.array(X.columns.astype(str))


def get_linear_model_coefficients(search, X: pd.DataFrame, top_n: int = 20):
    """
    Extract coefficients for Ridge / ElasticNet.
    """
    pipe = search.best_estimator_
    model = pipe.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError("This model does not expose linear coefficients.")

    feature_names = get_feature_names_from_pipeline(search, X)
    coef = model.coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    return coef_df.head(top_n)


def get_tree_feature_importance(search, X: pd.DataFrame, top_n: int = 20):
    """
    Extract impurity-based importance for tree models.
    """
    pipe = search.best_estimator_
    model = pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not expose feature importances.")

    feature_names = get_feature_names_from_pipeline(search, X)
    imp = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    return imp_df.head(top_n)


def run_ml_prediction(
    hmm_dir: Path,
    n_states: int,
    target: str,
):
    """
    Full ML workflow for one target.
    target: 'sym_s3' or 'sym_last'
    """
    Xy = build_ml_dataset(hmm_dir, n_states)
    X, y, d = get_ml_features_and_target(Xy, target=target)

    print(f"\n=== ML prediction for {target} ===")
    print(f"N patients: {len(d)}")
    print(f"N predictors before preprocessing: {X.shape[1]}")

    results_df, fitted_searches = evaluate_models_nested_cv(X, y)

    print("\nNested CV performance:")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_search = fitted_searches[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print("Best parameters:")
    print(best_search.best_params_)

    if best_model_name in ["ridge", "elasticnet"]:
        print("\nTop coefficients:")
        print(get_linear_model_coefficients(best_search, X, top_n=20))
    else:
        print("\nTop feature importances:")
        print(get_tree_feature_importance(best_search, X, top_n=20))

    return {
        "Xy": Xy,
        "X": X,
        "y": y,
        "results": results_df,
        "models": fitted_searches,
        "best_model_name": best_model_name,
    }