from modeling.model_base import BaseModel
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel(BaseModel):
  def __init__(self):
    pass
    self.versions = {
      "base": self.__compile_lstm_v1
    }
    
  def prepare(self, X):
    return X
  
  def compile(self, input_shape=128, output_shape=1, v="base"):
    if v not in self.versions:
      raise ValueError(f"Version '{v}' not recognized.")
    return self.versions[v](input_shape, output_shape)
  
  def __compile_lstm_v1(self, input_shape, output_shape):
    """https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-lstm-rnn-in-tensorflow/"""
    model = Sequential(
      [
        Input(shape=input_shape),
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),
        LSTM(units=128),
        Dropout(0.2),
        Dense(output_shape)
      ]
    )
    
    model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
    
    return model