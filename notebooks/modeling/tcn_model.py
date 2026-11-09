import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
  from modeling.model_base import BaseModel
else:
  from model_base import BaseModel
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Cropping1D,
    Dense,
    Dropout,
    Flatten,
    LayerNormalization,
    SpatialDropout1D,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class TCNModel(BaseModel):
  """
  Temporal Convolutional Network (Bai et al., 2018) para pct_posible_cola.

  task="classification" -> hay_cola (pct_posible_cola > threshold), salida sigmoide + BCE.
  task="regression"     -> pct_posible_cola (0 a 1), salida sigmoide + MSE.

  Flujo esperado:
      tcn = TCNModel(task="classification", window=48)
      X_seq, y_seq, meta = tcn.prepare(df.drop(columns="pct_posible_cola"), df["pct_posible_cola"])
      # ... TimeSeriesSplit sobre X_seq (ya viene ordenado cronologicamente por el bloque objetivo)
      X_tr, X_te = tcn.scale(X_tr, X_te)       # el scaler se ajusta SOLO con train
      model = tcn.compile(input_shape=X_tr.shape[1:], v="base")
      history = model.fit(X_tr, y_tr, epochs=50, batch_size=64, validation_split=0.1, callbacks=[...])
      df_pred, df_metrics = tcn.evaluate(model, history, X_te, y_te)
  """

  def __init__(self, task="classification", window=48, threshold=0.5):
    if task not in ("classification", "regression"):
      raise ValueError("task debe ser 'classification' o 'regression'.")
    self.task = task
    self.window = window
    self.threshold = threshold
    self.scaler = None
    self.feature_names_ = None
    self.versions = {
      "base": self.__compile_tcn_v1,
      "dense": self.__compile_tcn_v2,
      "norm": self.__compile_tcn_v3,
    }

  # ------------------------------------------------------------------ #
  # Preparacion de datos
  # ------------------------------------------------------------------ #
  def prepare(
      self,
      X,
      y=None,
      window=None,
      shift_target=True,
      ms_col="ms",
      time_col="bloque_tiempo",
      step_minutes=30,
      check_continuity=True,
      feature_cols=None,
      meta_extra=("hubo_ordenios",),
  ):
    """
    Arma ventanas 3D por robot: X_seq con forma (N, window, F).

    Parameters
    ----------
    X : pd.DataFrame
        Debe incluir las columnas `ms` y `bloque_tiempo`. El resto de columnas
        (o `feature_cols`) son las entradas. `temporada` (texto) se codifica
        en one-hot con categorias fijas; si existe `hora` se agregan hora_sin/hora_cos.
    y : array-like o pd.Series, opcional
        Objetivo alineado por posicion con X (pct_posible_cola).
    shift_target : bool, default=True
        True  -> la ventana termina en el bloque t-1 y la etiqueta es la del bloque t
                 (obligatorio con el dataset de serie de tiempo para no filtrar informacion).
        False -> la ventana termina en t. Usar solo si X no tiene variables mecanicas
                 del mismo bloque (p. ej. dataset de regresion).
    check_continuity : bool
        Descarta ventanas que cruzan huecos de tiempo (necesario en el dataset de
        regresion, que tiene huecos entre bloques).

    Returns
    -------
    X_seq : np.ndarray float32, (N, window, F)
    y_seq : np.ndarray float32, (N,) o None si no se paso y
    meta  : pd.DataFrame con ms, bloque_tiempo (del bloque objetivo) y meta_extra.
            Todo viene ordenado cronologicamente por el bloque objetivo, listo para
            TimeSeriesSplit.
    """
    window = window or self.window
    if not isinstance(X, pd.DataFrame):
      raise TypeError("X debe ser un DataFrame con las columnas 'ms' y 'bloque_tiempo'.")
    for c in (ms_col, time_col):
      if c not in X.columns:
        raise ValueError(f"Falta la columna '{c}' en X.")

    df = X.copy().reset_index(drop=True)
    if y is not None:
      df["__y__"] = np.asarray(y, dtype=float).ravel()
    df[time_col] = pd.to_datetime(df[time_col])

    # Codificaciones sin fuga (no dependen de train/test)
    if "temporada" in df.columns and df["temporada"].dtype == object:
      cats = ["invierno", "primavera", "verano", "otoño"]
      vals = df["temporada"].str.lower().str.strip()
      unknown = set(vals.dropna().unique()) - set(cats)
      if unknown:
        raise ValueError(f"Valores de temporada no reconocidos: {unknown}")
      dummies = pd.get_dummies(pd.Categorical(vals, categories=cats), prefix="temp").astype(np.float32)
      dummies.index = df.index
      df = pd.concat([df.drop(columns="temporada"), dummies], axis=1)
    if "hora" in df.columns:
      df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
      df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

    excluded = {ms_col, time_col, "__y__"}
    if feature_cols is None:
      feature_cols = [c for c in df.columns if c not in excluded]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
      raise ValueError(f"Columnas no numericas en las entradas: {non_numeric}")
    self.feature_names_ = list(feature_cols)

    meta_cols = [c for c in meta_extra if c in df.columns]
    step = np.timedelta64(step_minutes, "m")

    Xs, ys, metas = [], [], []
    for robot, g in df.sort_values([ms_col, time_col]).groupby(ms_col, sort=False):
      n = len(g)
      if n <= window:
        continue
      feats = g[feature_cols].to_numpy(dtype=np.float32)
      times = g[time_col].to_numpy()

      # (n-W+1, F, W) -> (n-W+1, W, F)
      wins = sliding_window_view(feats, window, axis=0).transpose(0, 2, 1)

      if shift_target:
        start = np.arange(0, n - window)
        tgt = start + window
      else:
        start = np.arange(0, n - window + 1)
        tgt = start + window - 1

      valid = np.ones(len(start), dtype=bool)
      if check_continuity:
        gap = np.r_[0, (np.diff(times) != step).astype(np.int64)]
        cs = np.cumsum(gap)
        valid &= (cs[tgt] - cs[start]) == 0

      if y is not None:
        y_vals = g["__y__"].to_numpy()[tgt]
        valid &= ~np.isnan(y_vals)

      start, tgt = start[valid], tgt[valid]
      if len(start) == 0:
        continue

      Xs.append(wins[start])
      if y is not None:
        ys.append(g["__y__"].to_numpy()[tgt])
      m = pd.DataFrame({ms_col: robot, time_col: times[tgt]})
      for c in meta_cols:
        m[c] = g[c].to_numpy()[tgt]
      metas.append(m)

    if not Xs:
      raise ValueError("No se pudo armar ninguna ventana; revisa window y la continuidad.")

    X_seq = np.concatenate(Xs, axis=0).astype(np.float32)
    meta = pd.concat(metas, ignore_index=True)
    order = np.argsort(meta[time_col].to_numpy(), kind="stable")
    X_seq = X_seq[order]
    meta = meta.iloc[order].reset_index(drop=True)

    y_seq = None
    if y is not None:
      y_seq = np.concatenate(ys, axis=0)[order].astype(np.float32)
      if self.task == "classification":
        y_seq = (y_seq > self.threshold).astype(np.float32)

    return X_seq, y_seq, meta

  def scale(self, X_train, *others):
    """Ajusta imputer (mediana) + StandardScaler solo con train y lo aplica a train y a los demas conjuntos 3D."""
    n_feat = X_train.shape[-1]
    # Igual que BaseModeling: imputer (mediana) + StandardScaler, pero ajustado solo con train.
    # La imputacion es necesaria: produccion_promedio es NaN en los bloques sin ordenios.
    self.scaler = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]).fit(X_train.reshape(-1, n_feat))

    def _tr(A):
      return self.scaler.transform(A.reshape(-1, n_feat)).reshape(A.shape).astype(np.float32)

    out = (_tr(X_train),) + tuple(_tr(A) for A in others)
    return out if others else out[0]

  # ------------------------------------------------------------------ #
  # Arquitectura
  # ------------------------------------------------------------------ #
  @staticmethod
  def receptive_field(kernel_size, levels):
    """RF = 1 + 2(k-1)(2^n - 1) para bloques con dos convoluciones (Bai et al., 2018)."""
    return 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)

  def _auto_levels(self, window, kernel_size):
    n = 1
    while self.receptive_field(kernel_size, n) < window:
      n += 1
    return n

  def compile(self, input_shape=(48, 20), output_shape=1, v="base", levels=None):
    """
    input_shape : (window, n_features), p. ej. X_train.shape[1:]
    levels      : niveles de dilatacion (d = 2^i). Si es None se elige el minimo
                  que cubra la ventana con el campo receptivo.
    """
    if v not in self.versions:
      raise ValueError(f"Version '{v}' not recognized.")
    if len(input_shape) != 2:
      raise ValueError("input_shape debe ser (window, n_features).")
    return self.versions[v](input_shape, output_shape, levels)

  def __residual_block(self, x, filters, kernel_size, dilation, dropout, use_norm):
    shortcut = x
    for _ in range(2):
      x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
      if use_norm:
        x = LayerNormalization()(x)
      x = Activation("relu")(x)
      x = SpatialDropout1D(dropout)(x)
    if shortcut.shape[-1] != filters:
      shortcut = Conv1D(filters, 1, padding="same")(shortcut)
    x = Add()([x, shortcut])
    return Activation("relu")(x)

  def __build(self, input_shape, output_shape, levels, filters, kernel_size,
              dropout, use_norm, head):
    window = input_shape[0]
    levels = levels or self._auto_levels(window, kernel_size)

    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(levels):
      x = self.__residual_block(x, filters, kernel_size, 2 ** i, dropout, use_norm)

    # Con convoluciones causales, el ultimo paso ya resume toda la ventana
    x = Cropping1D(cropping=(window - 1, 0))(x)
    x = Flatten()(x)
    for units in head:
      x = Dense(units, activation="relu")(x)
      x = Dropout(dropout)(x)
    outputs = Dense(output_shape, activation="sigmoid")(x)

    model = Model(inputs, outputs, name=f"tcn_k{kernel_size}_n{levels}")

    if self.task == "classification":
      model.compile(
          optimizer="adam",
          loss="binary_crossentropy",
          metrics=[
              "accuracy",
              tf.keras.metrics.AUC(name="auc"),
              tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall"),
          ],
      )
    else:
      model.compile(
          optimizer="adam",
          loss="mse",
          metrics=[
              tf.keras.metrics.MeanAbsoluteError(name="mae"),
              tf.keras.metrics.RootMeanSquaredError(name="rmse"),
          ],
      )
    return model

  def __compile_tcn_v1(self, input_shape, output_shape, levels):
    """Base: k=3, 64 filtros, cabeza densa pequena (Bai et al., 2018)."""
    return self.__build(input_shape, output_shape, levels, filters=64,
                        kernel_size=3, dropout=0.2, use_norm=False, head=[32])

  def __compile_tcn_v2(self, input_shape, output_shape, levels):
    """Dense: kernel mas ancho (k=5) y cabeza densa mas profunda."""
    return self.__build(input_shape, output_shape, levels, filters=64,
                        kernel_size=5, dropout=0.2, use_norm=False, head=[64, 32])

  def __compile_tcn_v3(self, input_shape, output_shape, levels):
    """Norm: como base pero con LayerNormalization en cada convolucion."""
    return self.__build(input_shape, output_shape, levels, filters=64,
                        kernel_size=3, dropout=0.2, use_norm=True, head=[32])

  # ------------------------------------------------------------------ #
  # Evaluacion
  # ------------------------------------------------------------------ #
  def evaluate(self, model, history, X_test, y_test, threshold=0.5):
    """
    Evalua el modelo segun self.task, imprime metricas y grafica.

    classification: Accuracy, Precision, Recall, F1, ROC-AUC + curvas, matriz de confusion y ROC.
    regression:     MAE, RMSE, R2 + curvas, real vs predicho y residuales.

    Returns
    -------
    df_predictions, df_summary_metrics
    """
    y_true = np.asarray(y_test).ravel()
    y_probs = model.predict(X_test, verbose=0).ravel()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    h = history.history

    if self.task == "classification":
      y_true = y_true.astype(int)
      y_pred = (y_probs >= threshold).astype(int)

      acc = accuracy_score(y_true, y_pred)
      prec = precision_score(y_true, y_pred, zero_division=0)
      rec = recall_score(y_true, y_pred, zero_division=0)
      f1 = f1_score(y_true, y_pred, zero_division=0)
      try:
        auc = roc_auc_score(y_true, y_probs)
      except ValueError:
        auc = np.nan

      df_summary_metrics = pd.DataFrame({
          "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
          "Value": [acc, prec, rec, f1, auc],
      })
      df_predictions = pd.DataFrame({
          "Actual": y_true,
          "Predicted Probability": y_probs,
          "Predicted Class": y_pred,
          "Correct": y_true == y_pred,
      })

      axes[0, 0].plot(h["loss"], label="Train Loss", linewidth=2)
      if "val_loss" in h:
        axes[0, 0].plot(h["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
      axes[0, 0].set_title("Binary Crossentropy Progression", fontsize=12, fontweight="bold")

      if "accuracy" in h:
        axes[0, 1].plot(h["accuracy"], label="Train Accuracy", color="green", linewidth=2)
        if "val_accuracy" in h:
          axes[0, 1].plot(h["val_accuracy"], label="Val Accuracy", color="green", linestyle="--", linewidth=2)
      if "auc" in h:
        axes[0, 1].plot(h["auc"], label="Train AUC", color="orange", linewidth=2)
        if "val_auc" in h:
          axes[0, 1].plot(h["val_auc"], label="Val AUC", color="orange", linestyle="--", linewidth=2)
      axes[0, 1].set_title("Accuracy & AUC Progression", fontsize=12, fontweight="bold")

      cm = confusion_matrix(y_true, y_pred)
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False)
      axes[1, 0].set_title(f"Confusion Matrix (Threshold = {threshold})", fontsize=12, fontweight="bold")
      axes[1, 0].set_xlabel("Predicted Class")
      axes[1, 0].set_ylabel("Actual Class")

      fpr, tpr, _ = roc_curve(y_true, y_probs)
      axes[1, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
      axes[1, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
      axes[1, 1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
      axes[1, 1].set_xlabel("False Positive Rate")
      axes[1, 1].set_ylabel("True Positive Rate")
      axes[1, 1].legend(loc="lower right")

      print("\n=== TEST SET CLASSIFICATION METRICS ===")

    else:
      y_true = y_true.astype(float)
      mae = mean_absolute_error(y_true, y_probs)
      rmse = float(np.sqrt(mean_squared_error(y_true, y_probs)))
      r2 = r2_score(y_true, y_probs)

      df_summary_metrics = pd.DataFrame({
          "Metric": ["MAE", "RMSE", "R2"],
          "Value": [mae, rmse, r2],
      })
      df_predictions = pd.DataFrame({
          "Actual": y_true,
          "Predicted": y_probs,
          "Residual": y_true - y_probs,
      })

      axes[0, 0].plot(h["loss"], label="Train MSE", linewidth=2)
      if "val_loss" in h:
        axes[0, 0].plot(h["val_loss"], label="Val MSE", linestyle="--", linewidth=2)
      axes[0, 0].set_title("MSE Progression", fontsize=12, fontweight="bold")

      for key, color in (("mae", "green"), ("rmse", "orange")):
        if key in h:
          axes[0, 1].plot(h[key], label=f"Train {key.upper()}", color=color, linewidth=2)
          if f"val_{key}" in h:
            axes[0, 1].plot(h[f"val_{key}"], label=f"Val {key.upper()}", color=color, linestyle="--", linewidth=2)
      axes[0, 1].set_title("MAE & RMSE Progression", fontsize=12, fontweight="bold")

      axes[1, 0].scatter(y_true, y_probs, s=4, alpha=0.3)
      axes[1, 0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
      axes[1, 0].set_title("Actual vs Predicted pct_posible_cola", fontsize=12, fontweight="bold")
      axes[1, 0].set_xlabel("Actual")
      axes[1, 0].set_ylabel("Predicted")

      axes[1, 1].hist(df_predictions["Residual"], bins=50, color="steelblue")
      axes[1, 1].set_title("Residuals", fontsize=12, fontweight="bold")
      axes[1, 1].set_xlabel("Actual - Predicted")

      print("\n=== TEST SET REGRESSION METRICS ===")

    print(df_summary_metrics.to_string(index=False))

    for ax in (axes[0, 0], axes[0, 1]):
      ax.set_xlabel("Epoch")
      ax.legend()
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Metric Value")

    plt.tight_layout()
    plt.show()

    return df_predictions, df_summary_metrics
