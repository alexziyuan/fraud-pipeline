from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlalchemy
import pickle
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

default_args = {
    'owner': 'fraud_pipeline',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DB_CONN = 'postgresql+psycopg2://fraud_user:fraud_pass@postgres/fraud_db'
MODEL_DIR = '/opt/airflow/data/models'
FEATURE_COLS = [
    'type', 'amount', 'old_balance_orig', 'new_balance_orig',
    'old_balance_dest', 'new_balance_dest', 'balance_diff_orig',
    'balance_diff_dest', 'orig_zero_start', 'orig_zero_end',
    'is_high_amount', 'account_tx_count', 'account_cashout_count'
]
TARGET_COL = 'is_fraud'


def load_features(**context):
    engine = sqlalchemy.create_engine(DB_CONN)
    logging.info("Loading sampled features...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    query = """
        SELECT * FROM v_features WHERE is_fraud = 1
        UNION ALL
        SELECT * FROM (
            SELECT * FROM v_features
            WHERE is_fraud = 0
            ORDER BY RANDOM()
            LIMIT 200000
        ) non_fraud
    """

    df = pd.read_sql(query, engine)
    logging.info(f"Loaded {len(df)} rows. Fraud rate: {df['is_fraud'].mean():.4f}")

    context['ti'].xcom_push(key='feature_shape', value=list(df.shape))
    context['ti'].xcom_push(key='fraud_rate', value=float(df['is_fraud'].mean()))

    df.to_parquet(f'{MODEL_DIR}/features.parquet', index=False)
    return {'rows': len(df)}

# Helper function to extract shared prep logic
def prepare_data():
    """Load parquet, encode type column, split train/test."""
    df = pd.read_parquet(f'{MODEL_DIR}/features.parquet')

    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, le


def compute_metrics(model, X_test, y_test):
    """Compute standard classification metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        'roc_auc':   round(roc_auc_score(y_test, y_proba), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

def log_model_run(engine, context, algorithm, metrics, model_path):
    """Write run metadata to model_runs table."""
    fraud_rate = context['ti'].xcom_pull(key='fraud_rate', task_ids='load_features')
    training_rows = context['ti'].xcom_pull(key='feature_shape', task_ids='load_features')[0]

    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO model_runs
            (run_id, model_version, training_rows, fraud_rate,
            roc_auc, precision_score, recall_score, f1_score,
            model_path, algorithm)
            VALUES
            (:run_id, :version, :rows, :fraud_rate,
             :roc_auc, :precision, :recall, :f1,
             :path, :algorithm)
        """), {
            'run_id': context['run_id'],
            'version': os.path.basename(model_path).replace('.pkl', ''),
            'rows': training_rows,
            'fraud_rate': fraud_rate,
            'roc_auc': metrics['roc_auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'path': model_path,
            'algorithm': algorithm,
        })


def train_logreg(**context):
    X_train, X_test, y_train, y_test, le = prepare_data()
    engine = sqlalchemy.create_engine(DB_CONN)

    # Apply SMOTE to training set only
    logging.info(f"Before SMOTE: fraud={y_train.sum()}, legit={(y_train == 0).sum()}")
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    logging.info(f"After SMOTE: fraud={y_train_sm.sum()}, legit={(y_train_sm == 0).sum()}")
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_sm, y_train_sm)

    metrics = compute_metrics(model, X_test, y_test)
    logging.info(f"LogReg Metrics: {metrics}")

    model_path = f'{MODEL_DIR}/logreg_{context["run_id"][:8]}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'encoder': le,
                     'features': FEATURE_COLS, 'algorithm': 'logreg'}, f)

    log_model_run(engine, context, 'logreg', metrics, model_path)
    context['ti'].xcom_push(key='logreg_auc', value=metrics['roc_auc'])


def train_random_forest(**context):
    X_train, X_test, y_train, y_test, le = prepare_data()
    engine = sqlalchemy.create_engine(DB_CONN)

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train_sm, y_train_sm)

    metrics = compute_metrics(model, X_test, y_test)
    logging.info(f"Random Forest Metrics: {metrics}")

    model_path = f"{MODEL_DIR}/rf_{context['run_id'][:8]}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'encoder': le,
                     'features': FEATURE_COLS, 'algorithm': 'random_forest'}, f)
    
    log_model_run(engine, context, 'random_forest', metrics, model_path)
    context['ti'].xcom_push(key='rf_auc', value=metrics['roc_auc'])

def train_xgboost(**context):
    X_train, X_test, y_train, y_test, le = prepare_data()
    engine = sqlalchemy.create_engine(DB_CONN)

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f"XGBoost with scale_pos_weight: {fraud_ratio:.1f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    metrics = compute_metrics(model, X_test, y_test)
    logging.info(f"XGBoost Metrics: {metrics}")

    model_path = f"{MODEL_DIR}/xgb_{context['run_id'][:8]}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'encoder': le,
                     'features': FEATURE_COLS, 'algorithm': 'xgboost'}, f)
        
    log_model_run(engine, context, 'xgboost', metrics, model_path)
    context['ti'].xcom_push(key='xgb_auc', value=metrics['roc_auc'])
    
def compare_and_promote(**context):
    ti = context['ti']
    engine = sqlalchemy.create_engine(DB_CONN)

    candidates = {
    'logreg':        ti.xcom_pull(key='logreg_auc', task_ids='train_logreg'),
    'random_forest': ti.xcom_pull(key='rf_auc', task_ids='train_random_forest'),
    'xgboost':       ti.xcom_pull(key='xgb_auc', task_ids='train_xgboost'),
    }

    logging.info(f"Model comparison: {candidates}")
    champion_algo = max(candidates, key=candidates.get)
    champion_auc = candidates[champion_algo]
    logging.info(f"Champion: {champion_algo} with ROC-AUC {champion_auc}")

    # Find the model file for the winning algorithm
    prefix_map = {
        'logreg': 'logreg',
        'random_forest': 'rf',
        'xgboost': 'xgb',
    }
    run_prefix = context['run_id'][:8]
    source_path = f'{MODEL_DIR}/{prefix_map[champion_algo]}_{run_prefix}.pkl'
    champion_path = f'{MODEL_DIR}/champion.pkl'

    import shutil
    shutil.copy2(source_path, champion_path)
    logging.info(f"champion.pkl updated from {source_path}")

    # Mark winning run in model_runs
    with engine.begin() as conn:
        # Clear previous champions
        conn.execute(sqlalchemy.text(
            "UPDATE model_runs SET is_champion = FALSE WHERE is_champion = TRUE"
        ))
        # Mark new champion
        conn.execute(sqlalchemy.text("""
            UPDATE model_runs SET is_champion = TRUE
            WHERE algorithm = :algo
              AND run_id = :run_id
        """), {'algo': champion_algo, 'run_id': context['run_id']})

    return {'champion': champion_algo, 'roc_auc': champion_auc}

with DAG(
    dag_id='train_fraud_model',
    default_args=default_args,
    description='Multi-model training pipeline with champion selection',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'fraud'],
) as dag:

    load_task = PythonOperator(task_id='load_features', python_callable=load_features)
    logreg_task = PythonOperator(task_id='train_logreg', python_callable=train_logreg)
    rf_task = PythonOperator(task_id='train_random_forest',python_callable=train_random_forest)
    xgb_task = PythonOperator(task_id='train_xgboost', python_callable=train_xgboost)
    promote_task = PythonOperator(task_id='compare_and_promote',python_callable=compare_and_promote)

    # load -> all three training tasks in parallel -> promote
    load_task >> [logreg_task, rf_task, xgb_task] >> promote_task