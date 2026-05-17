# AGENT: demo_agent
"""
CardioGraph-RL REST API.
Çalıştırmak için (cardiograph/ dizininden):
    uvicorn demo.api:app --port 8000 --reload
"""
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from demo.pipeline import CLASS_NAMES, run_inference

app = FastAPI(
    title="CardioGraph-RL API",
    description="EKG aritmi tespiti — GAT + Nöro-Sembolik Füzyon | PTB-XL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    pred_class: str
    pred_class_tr: str
    gat_probs: List[float]
    fused_probs: List[float]
    confidence_pct: float
    n_beats: int
    symbolic_rules: List[Dict[str, Any]]
    clinical_features: Dict[str, float]
    class_names: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": "CardioGAT", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    hea_file: UploadFile = File(..., description=".hea başlık dosyası"),
    dat_file: UploadFile = File(..., description=".dat sinyal dosyası"),
):
    stem = Path(hea_file.filename).stem

    with tempfile.TemporaryDirectory() as tmp:
        hea_path = Path(tmp) / f"{stem}.hea"
        dat_path = Path(tmp) / f"{stem}.dat"
        hea_path.write_bytes(await hea_file.read())
        dat_path.write_bytes(await dat_file.read())

        try:
            result = run_inference(str(hea_path.with_suffix("")))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference hatası: {exc}")

    pred_class = result["pred_class"]
    gat_probs  = result["gat_probs"].tolist()
    conf       = float(gat_probs[CLASS_NAMES.index(pred_class)])

    return PredictResponse(
        pred_class=pred_class,
        pred_class_tr=result["pred_class_tr"],
        gat_probs=gat_probs,
        fused_probs=result["fused_probs"].tolist(),
        confidence_pct=round(conf * 100, 2),
        n_beats=result["n_beats"],
        symbolic_rules=result["symbolic_rules"],
        clinical_features={k: round(float(v), 4) for k, v in result["clinical_features"].items()},
        class_names=CLASS_NAMES,
    )
