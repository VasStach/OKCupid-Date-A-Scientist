import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_score


def bucket_education(val: str) -> str | None:
    if pd.isna(val):
        return pd.NA
    val = str(val).lower()
    is_working_on = "working on" in val
    has_college_or_uni = any([exp in val for exp in ["college", "university", "masters program"]])
    has_phd_program = "ph.d program" in val
    has_graduated = "graduated" in val
    has_high_school = any(exp in val for exp in ["high school", "law school", "med school", "space camp"])
    has_dropped_out = "dropped out" in val

    if has_phd_program and not is_working_on and not has_dropped_out:
        return "Ph.D"

    if has_graduated and (has_college_or_uni or has_phd_program) or has_phd_program:
        return "graduated"

    if has_college_or_uni and not is_working_on and not has_dropped_out:
        return "graduated"

    if (has_high_school and not is_working_on and not has_dropped_out) or (
        (is_working_on or has_dropped_out) and has_college_or_uni
    ):
        return "high_school"

    return "lower"


def is_strict_diet(val: str) -> bool:
    if pd.isna(val):
        return False
    has_specific_diet = not any(exp in val for exp in ["anything", "other"])
    return True if "strictly" in val and has_specific_diet else False


def get_report(model: object, x_test: object, y_test: pd.Series) -> None:
    scores = cross_val_score(model, x_test, y_test, scoring="f1_weighted", cv=5)
    print("Macro-F1:", scores.mean())
    y_pred = model.predict(x_test)
    print(f1_score(y_test, y_pred, average="weighted"))
    unique_labels = y_test.value_counts().index.tolist()
    print(
        classification_report(
            y_test,
            y_pred,
            labels=unique_labels,
            target_names=["never", "sometimes", "often"],
            zero_division=0,
        )
    )


def get_best_thresholds(probs: np.ndarray, y_test: pd.Series) -> tuple[float, float]:
    prec_o, rec_o, thr_o = precision_recall_curve((y_test == 2), probs[:, 2])
    f1_o = 2 * (prec_o * rec_o) / (prec_o + rec_o + 1e-12)
    best_idx = np.argmax(f1_o)
    t_opt_o = thr_o[best_idx]
    print("Best 'often' thresh:", t_opt_o, "F1_often:", f1_o[best_idx])
    mask = probs[:, 2] <= t_opt_o
    prec_s, rec_s, thr_s = precision_recall_curve((y_test[mask] == 1), probs[mask, 1])
    f1_s = 2 * (prec_s * rec_s) / (prec_s + rec_s + 1e-12)
    best_idx = np.argmax(f1_s)
    t_opt_s = thr_s[best_idx]
    print("Best 'sometimes' thresh:", t_opt_s, "F1_sometimes:", f1_s[best_idx])
    return t_opt_o, t_opt_s


def get_best_thresh_binary(probs: np.ndarray, y_test: pd.Series) -> float:
    """
    Finds the optimal probability threshold for a binary classification problem
    (e.g., drugs vs. no drugs) based on the best F1 score.

    Args:
        probs (np.ndarray): Array of predicted probabilities for the positive class (shape: [n_samples,]).
        y_test (pd.Series): True binary labels (0 or 1).

    Returns:
        float: The threshold that yields the highest F1 score.
    """

    prec, rec, thresh = precision_recall_curve(y_test, probs)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = np.argmax(f1)
    t_opt = thresh[best_idx]
    print(f"Best threshold: {t_opt}, F1: {f1[best_idx]}")
    return t_opt


def predict_with_thresholds(probs: np.ndarray, t_o: float, t_s: float) -> np.ndarray:
    y_hat = np.full(len(probs), 0)
    mask_o = probs[:, 2] > t_o
    y_hat[mask_o] = 2
    mask_s = (~mask_o) & (probs[:, 1] > t_s)
    y_hat[mask_s] = 1
    return y_hat
