"""
API routes for model explainability
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from explainability.shap_explainer import DeepLogSHAPExplainer, AttentionExplainer
from explainability.visualization import (
    plot_shap_waterfall,
    plot_shap_summary,
    plot_attention_heatmap,
    create_explanation_report
)
from api.main import MODEL, MODEL_CONFIG
from api.database import get_db, Sequence
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])

# Global explainer instances
shap_explainer = None
attention_explainer = None


class ExplainRequest(BaseModel):
    sequence: List[int] = Field(..., description="Log sequence to explain")
    method: str = Field(default="shap", description="Explanation method: 'shap' or 'attention'")
    num_samples: int = Field(default=100, description="Number of samples for SHAP")


class ExplainBatchRequest(BaseModel):
    sequences: List[List[int]] = Field(..., description="List of sequences to explain")
    method: str = Field(default="shap", description="Explanation method")
    num_samples: int = Field(default=100, description="Number of samples for SHAP")


def get_shap_explainer():
    """Get or create SHAP explainer"""
    global shap_explainer
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if shap_explainer is None:
        shap_explainer = DeepLogSHAPExplainer(MODEL)
        # Prepare background data from database if available
        # For now, use empty background
    
    return shap_explainer


def get_attention_explainer():
    """Get or create attention explainer"""
    global attention_explainer
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if attention_explainer is None:
        attention_explainer = AttentionExplainer(MODEL)
    
    return attention_explainer


@router.post("/sequence")
async def explain_sequence(
    request: ExplainRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Explain a single sequence"""
    try:
        if request.method == "shap":
            explainer = get_shap_explainer()
            explanation = explainer.explain_sequence(
                request.sequence,
                num_samples=request.num_samples
            )
        elif request.method == "attention":
            explainer = get_attention_explainer()
            explanation = explainer.explain_sequence(request.sequence)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/batch")
async def explain_batch(
    request: ExplainBatchRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Explain multiple sequences"""
    try:
        if request.method == "shap":
            explainer = get_shap_explainer()
            explanations = explainer.explain_batch(
                request.sequences,
                num_samples=request.num_samples
            )
        elif request.method == "attention":
            explainer = get_attention_explainer()
            explanations = [
                explainer.explain_sequence(seq) for seq in request.sequences
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return {
            "explanations": explanations,
            "count": len(explanations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/visualize")
async def create_visualization(
    request: ExplainRequest,
    visualization_type: str = "waterfall"
) -> Dict[str, Any]:
    """Create visualization for explanation"""
    try:
        # Get explanation
        if request.method == "shap":
            explainer = get_shap_explainer()
            explanation = explainer.explain_sequence(
                request.sequence,
                num_samples=request.num_samples
            )
        else:
            explainer = get_attention_explainer()
            explanation = explainer.explain_sequence(request.sequence)
        
        # Create visualization
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if visualization_type == "waterfall" and request.method == "shap":
            save_path = str(output_dir / "shap_waterfall.png")
            plot_shap_waterfall(explanation, save_path)
        elif visualization_type == "attention" and request.method == "attention":
            save_path = str(output_dir / "attention_heatmap.png")
            plot_attention_heatmap(explanation, save_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Visualization type {visualization_type} not compatible with method {request.method}"
            )
        
        return {
            "explanation": explanation,
            "visualization_path": save_path,
            "visualization_type": visualization_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/report")
async def create_explanation_report_endpoint(
    request: ExplainRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Create text report for explanation"""
    try:
        if request.method == "shap":
            explainer = get_shap_explainer()
            explanation = explainer.explain_sequence(
                request.sequence,
                num_samples=request.num_samples
            )
        else:
            explainer = get_attention_explainer()
            explanation = explainer.explain_sequence(request.sequence)
        
        # Create report
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / "explanation_report.txt")
        
        report_text = create_explanation_report(explanation, save_path)
        
        return {
            "explanation": explanation,
            "report": report_text,
            "report_path": save_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report creation failed: {str(e)}")


@router.get("/summary")
async def get_explanation_summary(
    limit: int = 50,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get summary statistics for explanations"""
    try:
        # Get recent sequences from database
        sequences = db.query(Sequence).order_by(Sequence.created_at.desc()).limit(limit).all()
        
        if not sequences:
            return {
                "message": "No sequences found",
                "summary": {}
            }
        
        seq_list = [seq.sequence for seq in sequences]
        
        explainer = get_shap_explainer()
        summary_data = explainer.get_summary_plot_data(seq_list, max_samples=min(50, len(seq_list)))
        
        # Create summary plot
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / "shap_summary.png")
        
        plot_shap_summary(summary_data["explanations"], save_path)
        
        return {
            "summary_data": {
                "num_sequences": len(summary_data["explanations"]),
                "num_features": len(summary_data["features"][0]) if summary_data["features"] else 0
            },
            "visualization_path": save_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary creation failed: {str(e)}")
