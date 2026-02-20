"""
foundation_model utils/metrics.py
AUC 계산/예외 처리 유틸 (보조 역할).
실제 AUC는 RETFound lp engine 결과를 신뢰.
"""
import numpy as np
from typing import Optional


def safe_roc_auc_score(y_true, y_score, average: str = "macro") -> Optional[float]:
    """예외 처리된 ROC AUC 계산."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score, average=average))
    except (ValueError, TypeError):
        return None
