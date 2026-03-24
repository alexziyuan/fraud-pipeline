from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pickle
import os
import glob
import logging
import sqlalchemy
import pandas as pd
from datetime import datetime

# -- Logging ------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Config ------------
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DB_CONN = os.getenv("DB_CONN", "postgresql+psycopg2://fraud_user:fraud_pass@postgres/fraud_db")

# -- Global model state ------------
model_state = {}


def load_latest_model():
    """Load the champion model from MODEL_DIR."""
    champion_path = os.path.join(MODEL_DIR, "champion.pkl")
    if not os.path.exists(champion_path):
        raise FileNotFoundError(f"No champion model found at {champion_path}")
    with open(champion_path, "rb") as f:
        bundle = pickle.load(f)
    model_state["model"] = bundle["model"]
    model_state["encoder"] = bundle["encoder"]
    model_state["features"] = bundle["features"]
    model_state["version"] = "champion"
    model_state["algorithm"] = bundle.get("algorithm", "unknown")
    logger.info(f"Loaded champion model. Algorithm: {model_state['algorithm']}")

# -- Lifespan: load model at startup ------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_predictions_table()
    load_latest_model()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0",
    lifespan=lifespan,
)

# -- Pydantic schemas for input validation ------------
class TransactionRequest(BaseModel):
    type:              str     = Field(..., example="TRANSFER", description="TRANSFER or CASH_OUT")
    amount:            float   = Field(..., example=50000.0,    description="Transaction amount")
    old_balance_orig:  float   = Field(..., example=50000.0)
    new_balance_orig:  float   = Field(..., example=0.0)
    old_balance_dest:  float   = Field(..., example=0.0)
    new_balance_dest:  float   = Field(..., example=50000.0)
    account_tx_count:  int     = Field(1,   example=5,          description="Historical tx count for account")
    account_cashout_count: int = Field(0,   example=2)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold_used:    float
    model_version:     str
    scored_at:         str

# -- Prediction table ------------
def ensure_predictions_table():
    import time
    engine = sqlalchemy.create_engine(DB_CONN)
    for attempt in range(5):
        try:
            with engine.begin() as conn:
                conn.execute(sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id                SERIAL PRIMARY KEY,
                        type              VARCHAR(20),
                        amount            NUMERIC(18,2),
                        old_balance_orig  NUMERIC(18,2),
                        new_balance_orig  NUMERIC(18,2),
                        old_balance_dest  NUMERIC(18,2),
                        new_balance_dest  NUMERIC(18,2),
                        fraud_probability NUMERIC(6,4),
                        is_fraud          BOOLEAN,
                        model_version     VARCHAR(50),
                        scored_at         TIMESTAMP DEFAULT NOW()
                    )
                """))
            logger.info("Predictions table ready.")
            return
        except Exception as e:
            logger.warning(f"DB not ready (attempt {attempt+1}/5): {e}")
            time.sleep(3)
    raise RuntimeError("Could not connect to database after 5 attempts")

# -- Feature engineering (mirrors SQL view logic) ------------
def build_features(req: TransactionRequest) -> pd.DataFrame:
    le = model_state["encoder"]
    features = model_state["features"]
    type_encoded = le.transform([req.type])[0]

    row = {
        "type":             type_encoded,
        "amount":           req.amount,
        "balance_diff_orig": req.old_balance_orig - req.new_balance_orig,
        "orig_zero_end":    int(req.new_balance_orig == 0),
        "new_balance_dest": req.new_balance_dest,
    }
    return pd.DataFrame([row])[features]


# -- Routes ------------
@app.get("/health")
def health():
    """Liveness check - confirms API is running and model is loaded."""
    return {
        "status":        "healthy",
        "model_version": model_state.get("version", "not loaded"),
        }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """Score a single transaction for fraud probablity."""
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if transaction.type not in ["TRANSFER", "CASH_OUT"]:
        raise HTTPException(status_code=400,
                            detail="Only TRANSFER and CASH_OUT types are supported"
        )
    
    try:
        X = build_features(transaction)
        proba = float(model_state["model"].predict_proba(X)[0][1])
        threshold = model_state.get('threshold', 0.5)
        is_fraud = proba >= threshold

        # Persist prediction to DB
        engine = sqlalchemy.create_engine(DB_CONN)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("""
                INSERT INTO predictions 
                    (
                    type, amount, old_balance_orig, new_balance_orig,
                    old_balance_dest, new_balance_dest, fraud_probability,
                    is_fraud, model_version)
                VALUES
                    (:type, :amount, :old_orig, :new_orig,
                    :old_dest, :new_dest,
                    :prob, :fraud, :version)
            """), {
                    "type":     transaction.type,
                    "amount":   transaction.amount,
                    "old_orig": transaction.old_balance_orig,
                    "new_orig": transaction.new_balance_orig,
                    "old_dest": transaction.old_balance_dest,
                    "new_dest": transaction.new_balance_dest,
                    "prob":     proba,
                    "fraud":    is_fraud,
                    "version":  model_state["version"],
            })
        
        return PredictionResponse(
            fraud_probability=round(proba,4),
            is_fraud=is_fraud,
            threshold_used = threshold,
            model_version=model_state["version"],
            scored_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    """Hot-reload the latest model without restarting the container."""
    try:
        load_latest_model()
        return {"status": "reloaded", "model_version": model_state["version"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        