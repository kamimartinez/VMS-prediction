import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model_base import BaseModel
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

class LogisticRegression(BaseModel):
  def __init__(self):
    pass
    self.versions = {
      "base": self.__compile_lr_v1,
    }
    
  def prepare(self, X):
    return (None, 128, X.shape[1])
  
  def compile(self, input_shape=128, output_shape=1, v="base"):
    if v not in self.versions:
      raise ValueError(f"Version '{v}' not recognized.")
    return self.versions[v](input_shape, output_shape)
  
  def __compile_lr_v1(self, input_shape, output_shape=1):
    model = Sequential(
          [
            Input(shape=input_shape),
            Dense(output_shape, activation="sigmoid")
          ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
          "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
        ],
    )
    
    return model

  def evaluate(self, model, history, X_test, y_test, threshold=0.5):
        """
        Evaluates the binary classification model on test data, prints classification performance,
        plots training history (Loss, Accuracy, AUC), Matrix of Confusion, and ROC Curve.
    
        Parameters:
        -----------
        model : tf.keras.Model
            The trained Keras model instance.
        history : tf.keras.callbacks.History
            The history object returned by model.fit().
        X_test : np.ndarray
            Prepared 3D test feature array.
        y_test : np.ndarray or pd.Series
            True target values for the test set (0 or 1).
        threshold : float, default=0.5
            Probability threshold to classify positive predictions (1).
    
        Returns:
        --------
        df_predictions : pd.DataFrame
            DataFrame with actuals, predicted probabilities, predicted classes, and correctness.
        df_summary_metrics : pd.DataFrame
            Summary table of classification evaluation metrics.
        """
        # Guarantee 1D integer target array
        y_true = np.asarray(y_test).ravel().astype(int)
    
        # Obtain predicted probabilities and convert to binary classes via threshold
        y_probs = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_probs >= threshold).astype(int)
    
        # Calculate Classification Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
        try:
            auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            auc = np.nan
    
        df_summary_metrics = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "Value": [acc, prec, rec, f1, auc],
            }
        )
    
        print("\n=== TEST SET CLASSIFICATION METRICS ===")
        print(df_summary_metrics.to_string(index=False))
    
        # Prediction dataframe for history.predict(test)
        df_predictions = pd.DataFrame(
            {
                "Actual": y_true,
                "Predicted Probability": y_probs,
                "Predicted Class": y_pred,
                "Correct": y_true == y_pred,
            }
        )
    
        # Setup plots
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
        # 1. Loss Progression (Binary Crossentropy)
        axes[0, 0].plot(history.history["loss"], label="Train Loss", linewidth=2)
        if "val_loss" in history.history:
            axes[0, 0].plot(history.history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
        axes[0, 0].set_title("Binary Crossentropy Progression", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
    
        # 2. Accuracy & AUC Progression
        acc_key = "accuracy" if "accuracy" in history.history else "acc"
        if acc_key in history.history:
            axes[0, 1].plot(history.history[acc_key], label="Train Accuracy", color="green", linewidth=2)
            if f"val_{acc_key}" in history.history:
                axes[0, 1].plot(history.history[f"val_{acc_key}"], label="Val Accuracy", color="green", linestyle="--", linewidth=2)
        if "auc" in history.history:
            axes[0, 1].plot(history.history["auc"], label="Train AUC", color="orange", linewidth=2)
            if "val_auc" in history.history:
                axes[0, 1].plot(history.history["val_auc"], label="Val AUC", color="orange", linestyle="--", linewidth=2)
        axes[0, 1].set_title("Accuracy & AUC Progression", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Metric Value")
        axes[0, 1].legend()
    
        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False)
        axes[1, 0].set_title(f"Confusion Matrix (Threshold = {threshold})", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Predicted Class")
        axes[1, 0].set_ylabel("Actual Class")
    
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        axes[1, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        axes[1, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[1, 1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("False Positive Rate")
        axes[1, 1].set_ylabel("True Positive Rate")
        axes[1, 1].legend(loc="lower right")
    
        plt.tight_layout()
        plt.show()
    
        return df_predictions, df_summary_metrics
      
  def print_weigths(self, model, X_test, columns):
    weights, bias = model.layers[0].get_weights()
    weights = weights.flatten()
    b = bias[0]

    # 1. Calcular las probabilidades predichas
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    # 2. Prevenir valores numéricos extremos (0.0 o 1.0) mediante clipping
    eps = 1e-8
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    p_diag = y_pred_prob * (1 - y_pred_prob)

    # 3. Construir matriz de diseño X con columna para el bias: [X_test, 1]
    X_design = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    # 4. Calcular la Matriz de Información de Fisher (X^T * V * X)
    V_X = X_design * p_diag[:, np.newaxis]
    fisher_info = np.dot(X_design.T, V_X)

    # 5. Usar Pseudoinversa (pinv) en lugar de inv para evitar Singular Matrix
    cov_matrix = np.linalg.pinv(fisher_info)

    # Errores estándar (se asegura que no haya valores negativos bajo la raíz por imprecisión numérica)
    std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))

    # 6. Z-scores y P-values
    all_weights = np.append(weights, b)
    
    # Manejar posibles errores estándar cero/nulos
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = np.where(std_errors > 0, all_weights / std_errors, 0.0)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # 7. Construir y desplegar el DataFrame
    feature_names = [i for i in columns] + ["Intercept (Bias)"]

    df_stats = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": all_weights,
        "Std Error": std_errors,
        "p-value": p_values
    })

    df_stats["Significativo (p < 0.05)"] = df_stats["p-value"] < 0.05
    print(df_stats.to_string(index=False))