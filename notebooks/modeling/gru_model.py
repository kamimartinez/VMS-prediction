import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
  from modeling.tcn_model import TCNModel
else:
  from tcn_model import TCNModel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


class GRUModel(TCNModel):
  """
  Red recurrente GRU (Cho et al., 2014) para pct_posible_cola.

  Reutiliza de TCNModel la preparacion de datos (ventanas 3D por robot, ventana
  en t-1 y etiqueta en t), el escalado ajustado solo con train y la evaluacion.
  Solo cambia la arquitectura.

  task="regression"     -> pct_posible_cola (0 a 1), salida sigmoide + MSE.  (default)
  task="classification" -> hay_cola (pct_posible_cola > threshold), salida sigmoide + BCE.

  Flujo esperado (split cronologico 70/15/15, igual que Diseno de Pruebas):
      gru = GRUModel(task="regression", window=12)
      X_seq, y_seq, meta = gru.prepare(df, df["pct_posible_cola"])
      tr, va, te = gru.split_temporal(df, meta)        # 70 / 15 / 15 por fila del dataset
      X_tr, X_va, X_te = gru.scale(X_seq[tr], X_seq[va], X_seq[te])   # scaler ajustado SOLO con train
      model = gru.compile(input_shape=X_tr.shape[1:], v="custom", units=32, layers=1)
      history = model.fit(X_tr, y_seq[tr], validation_data=(X_va, y_seq[va]),
                          epochs=50, batch_size=128, callbacks=[...])
      # el test (te) se usa una sola vez, con el modelo final
  """

  def __init__(self, task="regression", window=48, threshold=0.5, learning_rate=1e-3):
    super().__init__(task=task, window=window, threshold=threshold)
    self.learning_rate = learning_rate
    self.versions = {
      "base": self.__compile_gru_v1,
      "norm": self.__compile_gru_v2,
      "bi": self.__compile_gru_v3,
    }

  # ------------------------------------------------------------------ #
  # Arquitectura
  # ------------------------------------------------------------------ #
  def compile(self, input_shape=(48, 27), output_shape=1, v="base", **cfg):
    """
    input_shape : (window, n_features), p. ej. X_train.shape[1:]
    v           : "base" | "norm" | "bi" (versiones fijas) o "custom" (configurable,
                  para la escalera de experimentos; ver build_custom).
    **cfg       : solo con v="custom": units, layers, dropout, dense, norm, bidirectional.
    """
    if len(input_shape) != 2:
      raise ValueError("input_shape debe ser (window, n_features).")
    if v == "custom":
      model = self.build_custom(tuple(input_shape), output_shape, **cfg)
    elif v in self.versions:
      model = self.versions[v](tuple(input_shape), output_shape)
    else:
      raise ValueError(f"Version '{v}' not recognized.")
    self.__compile_model(model)
    return model

  def build_custom(self, input_shape, output_shape=1, units=32, layers=1, dropout=0.0,
                   dense=0, norm=False, bidirectional=False):
    """
    GRU configurable para la escalera de experimentos (de lo simple a lo complejo).

    units         : unidades de la primera capa GRU.
    layers        : capas GRU apiladas; cada capa extra tiene la mitad de unidades
                    (64 -> 32), siguiendo la reduccion progresiva de Das (2023).
    dropout       : Dropout despues de cada capa GRU (0 = sin dropout).
    dense         : neuronas de una capa Dense(ReLU) antes de la salida (0 = sin capa).
    norm          : LayerNormalization despues de cada capa GRU.
    bidirectional : la primera capa GRU es bidireccional. Es valido porque toda la
                    ventana es pasado (termina en t-1).

    E0 (GRU minima) : units=32, layers=1  -> GRU(32) -> Dense(1, sigmoid)
    Ref (modelo v1) : units=64, layers=2, dropout=0.2, dense=32  (igual a v="base")
    """
    if layers < 1:
      raise ValueError("layers debe ser >= 1.")
    name = f"gru_u{units}_l{layers}" + (f"_d{dropout:g}" if dropout else "") \
        + (f"_dense{dense}" if dense else "") + ("_norm" if norm else "") + ("_bi" if bidirectional else "")
    model = Sequential(name=name.replace(".", ""))
    model.add(Input(shape=input_shape))
    for i in range(layers):
      n = max(units // (2 ** i), 1)
      gru = GRU(n, return_sequences=(i < layers - 1))
      model.add(Bidirectional(gru) if (bidirectional and i == 0) else gru)
      if norm:
        model.add(LayerNormalization())
      if dropout > 0:
        model.add(Dropout(dropout))
    if dense > 0:
      model.add(Dense(dense, activation="relu"))
    model.add(Dense(output_shape, activation="sigmoid"))
    return model

  # ------------------------------------------------------------------ #
  # Split cronologico 70 / 15 / 15
  # ------------------------------------------------------------------ #
  def split_temporal(self, df, meta, train_frac=0.70, val_frac=0.15,
                     ms_col="ms", time_col="bloque_tiempo"):
    """
    Divide las ventanas en train / val / test igual que Diseno de Pruebas y que la LSTM:
    se ordena el dataset por (bloque_tiempo, ms) y se corta por fila en 70 % / 15 % / 15 %.
    Cada ventana se asigna al conjunto de la fila de su bloque OBJETIVO (t), por lo que
    el test queda con exactamente los mismos bloques que el de los demas modelos,
    sin importar el largo de la ventana. La ventana de un bloque de val/test puede traer
    historia del periodo anterior como entrada; eso no es fuga porque ya ocurrio.

    Returns
    -------
    tr, va, te : arreglos de indices sobre X_seq / y_seq / meta.
    """
    d = df[[ms_col, time_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values([time_col, ms_col], kind="stable").reset_index(drop=True)
    if d.duplicated([ms_col, time_col]).any():
      raise ValueError("Hay pares (ms, bloque_tiempo) repetidos en el dataset.")
    n = len(d)
    n_tr, n_va = int(n * train_frac), int(n * (train_frac + val_frac))
    d["__pos__"] = np.arange(n)

    m = meta[[ms_col, time_col]].copy()
    m[time_col] = pd.to_datetime(m[time_col])
    pos = m.merge(d, on=[ms_col, time_col], how="left")["__pos__"].to_numpy()
    if np.isnan(pos).any():
      raise ValueError("Hay ventanas cuyo bloque objetivo no esta en el dataset.")

    tr = np.where(pos < n_tr)[0]
    va = np.where((pos >= n_tr) & (pos < n_va))[0]
    te = np.where(pos >= n_va)[0]
    self.split_info_ = {
        "filas": {"train": n_tr, "val": n_va - n_tr, "test": n - n_va},
        "cortes": {"inicio_val": d[time_col].iloc[n_tr], "inicio_test": d[time_col].iloc[n_va]},
        "ventanas": {"train": len(tr), "val": len(va), "test": len(te)},
    }
    return tr, va, te

  def __compile_model(self, model):
    # clipnorm evita gradientes explosivos en redes recurrentes
    optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
    if self.task == "classification":
      model.compile(
          optimizer=optimizer,
          loss="binary_crossentropy",
          metrics=[
              "accuracy",
              tf.keras.metrics.AUC(name="auc"),
              tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
              tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall"),
          ],
      )
    else:
      model.compile(
          optimizer=optimizer,
          loss="mse",
          metrics=[
              tf.keras.metrics.MeanAbsoluteError(name="mae"),
              tf.keras.metrics.RootMeanSquaredError(name="rmse"),
          ],
      )

  # Nota: no se usa recurrent_dropout para conservar la implementacion cuDNN (GPU en Colab).
  def __compile_gru_v1(self, input_shape, output_shape):
    """Base: dos capas GRU apiladas (64 -> 32) con Dropout."""
    return Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(output_shape, activation="sigmoid"),
    ], name="gru_base")

  def __compile_gru_v2(self, input_shape, output_shape):
    """Norm: como base + LayerNormalization y regularizacion L2 en la cabeza."""
    return Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        LayerNormalization(),
        Dropout(0.3),
        GRU(32),
        LayerNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(output_shape, activation="sigmoid"),
    ], name="gru_norm")

  def __compile_gru_v3(self, input_shape, output_shape):
    """
    Bi: GRU bidireccional sobre la ventana. Es valido porque toda la ventana
    es pasado (termina en t-1); no ve el bloque que se predice.
    """
    return Sequential([
        Input(shape=input_shape),
        Bidirectional(GRU(48, return_sequences=True)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(output_shape, activation="sigmoid"),
    ], name="gru_bi")

  # ------------------------------------------------------------------ #
  # Utilidades para el desbalance de clases
  # ------------------------------------------------------------------ #
  def class_weights(self, y_train):
    """Pesos 'balanced' para model.fit(class_weight=...). Devuelve None en regresion."""
    if self.task != "classification":
      return None
    y = np.asarray(y_train).ravel().astype(int)
    classes = np.unique(y)
    if len(classes) < 2:
      return None
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(v) for c, v in zip(classes, w)}

  def best_threshold(self, model, X_val, y_val, grid=None):
    """
    Busca el umbral que maximiza F1 en un conjunto de VALIDACION (nunca en test).
    En regresion convierte y_val con self.threshold para definir hay_cola.
    Devuelve (umbral, f1).
    """
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    y_true = (np.asarray(y_val).ravel() > self.threshold).astype(int) \
        if self.task == "regression" else np.asarray(y_val).ravel().astype(int)
    probs = model.predict(X_val, verbose=0).ravel()
    scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])
