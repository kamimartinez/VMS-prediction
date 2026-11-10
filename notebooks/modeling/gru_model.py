import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
  from modeling.tcn_model import TCNModel
else:
  from tcn_model import TCNModel
import numpy as np
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

  Flujo esperado:
      gru = GRUModel(task="regression", window=48)
      X_seq, y_seq, meta = gru.prepare(df, df["pct_posible_cola"])
      # ... TimeSeriesSplit sobre X_seq (ya viene ordenado cronologicamente)
      X_tr, X_te = gru.scale(X_seq[tr], X_seq[te])     # scaler ajustado SOLO con train
      model = gru.compile(input_shape=X_tr.shape[1:], v="base")
      history = model.fit(X_tr, y_tr, epochs=50, batch_size=128, validation_split=0.1,
                          class_weight=gru.class_weights(y_tr),   # None en regresion
                          callbacks=[...])
      df_pred, df_metrics = gru.evaluate(model, history, X_te, y_te)
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
  def compile(self, input_shape=(48, 27), output_shape=1, v="base", **_):
    """input_shape : (window, n_features), p. ej. X_train.shape[1:]"""
    if v not in self.versions:
      raise ValueError(f"Version '{v}' not recognized.")
    if len(input_shape) != 2:
      raise ValueError("input_shape debe ser (window, n_features).")
    model = self.versions[v](tuple(input_shape), output_shape)
    self.__compile_model(model)
    return model

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
