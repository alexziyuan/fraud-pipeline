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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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


def train_model(**context):
    """Train XGBoost, evaluate, serialize model, log metrics to Postgres."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load features from disk
    df = pd.read_parquet(f'{MODEL_DIR}/features.parquet')

    # Encode categorical: 'type' is TRANSFER or CASH_OUT
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Stratified split to preserve fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weight to handle severe imbalance (~1% fraud rate)
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f"Class weight ratio: {fraud_ratio:.1f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,   # key param for imbalanced data
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'roc_auc':   round(roc_auc_score(y_test, y_pred_proba), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    logging.info(f"Metrics: {metrics}")

    # Serialize model + encoder together
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'{MODEL_DIR}/xgb_{model_version}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'encoder': le, 'features': FEATURE_COLS}, f)
    logging.info(f"Model saved to {model_path}")

    # Log to Postgres
    engine = sqlalchemy.create_engine(DB_CONN)
    fraud_rate = context['ti'].xcom_pull(key='fraud_rate', task_ids='load_features')
    training_rows = context['ti'].xcom_pull(key='feature_shape', task_ids='load_features')[0]

    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO model_runs
                (run_id, model_version, training_rows, fraud_rate,
                 roc_auc, precision_score, recall_score, f1_score, model_path)
            VALUES
                (:run_id, :version, :rows, :fraud_rate,
                 :roc_auc, :precision, :recall, :f1, :path)
        """), {
            'run_id':     context['run_id'],
            'version':    model_version,
            'rows':       training_rows,
            'fraud_rate': fraud_rate,
            'roc_auc':    metrics['roc_auc'],
            'precision':  metrics['precision'],
            'recall':     metrics['recall'],
            'f1':         metrics['f1'],
            'path':       model_path,
        })

    logging.info("Metrics logged to model_runs table.")


with DAG(
    dag_id='train_fraud_model',
    default_args=default_args,
    description='Feature engineering + XGBoost training for fraud detection',
    schedule_interval='@weekly',      # retrain weekly on fresh data
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'fraud'],
) as dag:

    load_task = PythonOperator(
        task_id='load_features',
        python_callable=load_features,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Dependency: load must complete before train starts
    load_task >> train_task