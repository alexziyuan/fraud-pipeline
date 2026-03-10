CREATE TABLE IF NOT EXISTS transactions_raw (
    step INTEGER,
    type VARCHAR(20),
    amount NUMERIC(18, 2),
    name_orig VARCHAR(50),
    old_balance_orig NUMERIC(18, 2),
    new_balance_orig NUMERIC(18, 2),
    name_dest VARCHAR(50),
    old_balance_dest NUMERIC(18, 2),
    new_balance_dest NUMERIC(18, 2),
    is_fraud INTEGER,
    is_flagged_fraud INTEGER,
    ingested_at TIMESTAMP DEFAULT NOW()
    );
    -- Ingestion audit log
    CREATE TABLE IF NOT EXISTS ingestion_log (
    id SERIAL PRIMARY KEY ,
    run_id VARCHAR(100),
    rows_loaded INTEGER,
    source_file VARCHAR(200),
    status VARCHAR(20),
    started_at TIMESTAMP,
    finished_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(100),
    model_version   VARCHAR(50),
    training_rows   INTEGER,
    fraud_rate      NUMERIC(6, 4),
    roc_auc         NUMERIC(6, 4),
    precision_score NUMERIC(6, 4),
    recall_score    NUMERIC(6, 4),
    f1_score        NUMERIC(6, 4),
    model_path      VARCHAR(200),
    trained_at      TIMESTAMP DEFAULT NOW()
);