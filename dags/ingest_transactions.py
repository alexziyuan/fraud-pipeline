from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
import os
import logging

default_args = {
    'owner': 'fraud_pipeline',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

DATA_PATH = '/opt/airflow/data/transactions_raw.csv'
DB_CONN = 'postgresql+psycopg2://fraud_user:fraud_pass@postgres/fraud_db'
CHUNK_SIZE = 50_000  # process in chunks to avoid memory issues


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if expected columns are missing."""
    required = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
                'newbalanceOrig', 'nameDest', 'oldbalanceDest',
                'newbalanceDest', 'isFraud', 'isFlaggedFraud']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    logging.info("Schema validation passed.")


def validate_data_quality(df: pd.DataFrame) -> None:
    """Basic data quality checks."""
    null_counts = df[['amount', 'isFraud']].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in critical columns: {null_counts}")
    if (df['amount'] < 0).any():
        raise ValueError("Negative transaction amounts detected.")
    logging.info(f"Data quality checks passed. Rows: {len(df)}")


def ingest_transactions(**context):
    run_id = context['run_id']
    engine = sqlalchemy.create_engine(DB_CONN)
    started_at = datetime.now()
    total_rows = 0

    logging.info(f"Starting ingestion. Run ID: {run_id}")

    # Read and validate first chunk for schema check
    first_chunk = pd.read_csv(DATA_PATH, nrows=100)
    validate_schema(first_chunk)

    # Truncate table before full reload (idempotent)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("TRUNCATE TABLE transactions_raw"))

    # Load in chunks
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        validate_data_quality(chunk)

        chunk = chunk.rename(columns={
            'nameOrig': 'name_orig',
            'oldbalanceOrg': 'old_balance_orig',
            'newbalanceOrig': 'new_balance_orig',
            'nameDest': 'name_dest',
            'oldbalanceDest': 'old_balance_dest',
            'newbalanceDest': 'new_balance_dest',
            'isFraud': 'is_fraud',
            'isFlaggedFraud': 'is_flagged_fraud',
        })

        chunk.to_sql('transactions_raw', engine, if_exists='append', index=False)
        total_rows += len(chunk)
        logging.info(f"Loaded {total_rows} rows so far...")

    # Write audit log
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO ingestion_log (run_id, rows_loaded, source_file, status, started_at)
            VALUES (:run_id, :rows, :source, 'success', :started)
        """), {"run_id": run_id, "rows": total_rows,
               "source": DATA_PATH, "started": started_at})
        
    logging.info(f"Ingestion complete. Total rows loaded: {total_rows}")
    return total_rows


with DAG(
    dag_id='ingest_transactions',
    default_args=default_args,
    description='Ingest raw PaySim transactions into Postgres',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ingestion', 'fraud'],
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_raw_transactions',
        python_callable=ingest_transactions,
    )