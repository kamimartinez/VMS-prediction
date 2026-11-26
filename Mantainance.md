# Plan de Monitoreo y Mantenimiento

### Concentración en Inteligencia Artificial Avanzada (Octubre - Diciembre 2026)

## 1. Introducción

El equipo responsable del proyecto de investigación durante su desarrollo en el periodo Octubre-Diciembre 2026 no dará mantenimiento continuo al sistema tras la finalización de la materia de concentración de Inteligencia Artificial Avanzada. Debido a esto, el presente plan de monitoreo repasa las acciones operativas necesarias para dar mantenimiento al proyecto y permitir su utilización por parte de cualquier persona interesada.

Es fundamental considerar que el proyecto fue desarrollado con datos del CAETEC (2022-2026), periodo en el cual se detectó un incremento en la cantidad de vacas del rancho, alcanzando un punto donde la ocupación de las máquinas de ordeño voluntario (VMS) se acerca a un estado crítico. Por ello, el proyecto podría perder viabilidad en el corto plazo si ocurren cambios estructurales en la cantidad de máquinas o en el tamaño del hato.
Objetivos del Proyecto y Modelos Generados
Los objetivos principales consisten en identificar las variables explicativas con mayor impacto en la formación de colas en las máquinas de ordeño voluntario. Los modelos desarrollados se dividen en dos categorías principales:

Modelos Descriptivos: Buscan entender la relación de las variables explicativas y su impacto al determinar si existe o no una cola (ej. Regresión Logística y MLP).
Modelos Predictivos: Buscan identificar patrones temporales en la formación de colas para predecir el comportamiento del sistema en momentos posteriores del día o con un día de anticipación.

Si estos propósitos no se adecúan al nuevo caso de uso requerido, se sugiere revisar los avances y resultados del proyecto para tomarlos como base en un nuevo ciclo de desarrollo.

## 2. Pasos de Mantenimiento y Replicación

Los siguientes pasos guían al usuario desde la clonación del repositorio en un ambiente local hasta la preparación de datos, entrenamiento del modelo y ejecución de pruebas.

Nota: Este repositorio no distribuye la información confidencial o sensible del CAETEC, por lo que para entrenar los modelos será necesario incluir una fuente de datos propia o adaptar el código a la nueva estructura de entrada.
Paso 1: Clonar el Repositorio y Configurar el Ambiente
Si se planea ejecutar el proyecto completamente en Google Colab, la configuración puede variar. Para ejecutarlo de manera local, utilice los siguientes comandos:h
Clonar repositorio
git clone https://github.com/kamimartinez/VMS-prediction
cd VMS-prediction
Crear y activar entorno virtual
uv venv venv
En Windows:
source ./venv/Scripts/activate
En Linux / Mac:
source ./venv/bin/activate
Instalar dependencias
pip install -r requirements.txt

_Nota: La instalación puede demorar algunos minutos debido a dependencias pesadas como TensorFlow._

### Paso 2: Preparación del Ambiente y Datos (Google Colab / Drive)

La fase de exploración y preparación de datos inicial se estructuró en Google Colab para procesar archivos directamente almacenados en Google Drive.

1.  Crear una carpeta en la raíz de Google Drive con la ruta: `/Datos/Crudo`.
2.  Colocar los archivos fuente en dicha carpeta.
3.  Dentro de la ruta `/notebooks/data` se encuentran los scripts de procesamiento:
    - `DataUnderstanding.ipynb`: Grafica distribuciones de ordeño y patrones detectados. Transforma los datos crudos a formatos más ligeros para optimizar lecturas posteriores.
    - `DataPreparation.ipynb`: Toma los datos procesados de la fase anterior y los estructura adecuadamente antes de ser ingresados a los modelos de ML/DL.

### Paso 3: Entrenamiento de los Modelos

Los modelos se encuentran organizados en la carpeta `/notebooks/modeling`. La estructura base de cada algoritmo consta de dos archivos:

- `<model>.py`: Contiene la implementación lógica y arquitectura del modelo.
- `<notebook>.ipynb`: Gestiona la carga de datos, transformaciones adicionales, partición del conjunto de datos (Train/Validation/Test) y la ejecución del entrenamiento.

Al finalizar el entrenamiento de un modelo, el estado compilado se exporta automáticamente como un archivo `.keras` dentro de su subcarpeta en `/models/` (por ejemplo, `/models/lstm/` o `/models/tcn/`), el cual es requerido para la etapa de validación.

### Paso 4: Ejecución de Pruebas y Evaluación

Una vez compilados los modelos y generado el split de datos (mediante `notebooks/modeling/create_test_split.py`), se ejecutan los scripts de evaluación ubicados en `notebooks/evaluation/`. Estos scripts evalúan los resultados del modelo frente a los objetivos de negocio y de minería de datos, permitiendo una comparativa directa del rendimiento entre arquitecturas.

---

## 3. Guía de Ejecución y Monitoreo Continuo

Al tratarse de un sistema basado en ciencia de datos e inteligencia artificial, los modelos perderán precisión con el tiempo debido a cambios en el comportamiento del hato, rotación de vacas o modificaciones operativas en el rancho.

### Frecuencia de Evaluación

Se recomienda realizar una evaluación de rendimiento **cada 3 a 6 meses**.

| Estado del Modelo          | Métrica F1-Score | Acción Recomendada                                                                                                    |
| -------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Óptimo**                 | 0.65 – 0.80      | Mantener modelo en producción sin modificaciones.                                                                     |
| **Obsoleto / Deteriorado** | < 0.60           | Reentrenar el modelo, ajustar hiperparámetros o descontinuar uso.                                                     |
| **Alerta de Saturación**   | > 0.85           | El modelo detecta una presencia crítica y constante de colas. Se sugiere intervención operativa directa en el rancho. |

### Criterios para Descontinuar o Reestructurar el Modelo

- Disminución del F1-Score por debajo de **0.65**.
- F1-Score persistentemente superior a **0.85** (indica cambio en la dinámica operativa y saturación de la infraestructura).
- Adición o retiro de máquinas de ordeño en la infraestructura física del rancho.

---

## 4. Adaptación de Modelos e Hiperparámetros

La mayoría de las modificaciones estructurales se realizan dentro de los archivos `.py` asociados a cada modelo. A continuación se describe la arquitectura estándar y los componentes adaptables.

### Estructura del Código del Modelo

La clase principal del modelo se divide en 4 secciones clave:

1.  **Definición de versiones:** Registro de arquitecturas disponibles en el constructor y método `compile()`.
2.  **Preparación de datos:** Método `prepare()` para transformar los datos al formato de entrada requerido.
3.  **Implementaciones de arquitectura:** Funciones privadas que definen las capas, hiperparámetros y métricas.
4.  **Evaluación:** Método `evaluate()` para despliegue de métricas estandarizadas.

```python
class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.versions = {
            "base": self.__compile_lstm_v1,
            "dense": self.__compile_lstm_v2,
            "norm": self.__compile_lstm_v3,
            "norm+drop": self.__compile_lstm_v4
        }

    def prepare(self, X):
        return X.reshape((X.shape[0], 1, X.shape[1]))

    def compile(self, input_shape=128, output_shape=1, v="base"):
        if v not in self.versions:
            raise ValueError(f"Version '{v}' not recognized.")
        return self.versions[v](input_shape, output_shape)

    def __compile_lstm_v1(self, input_shape, output_shape):
        """Basado en la arquitectura LSTM en TensorFlow"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=128, return_sequences=True),
            Dropout(0.2),
            LSTM(units=128),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(output_shape, activation='sigmoid')
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
        # Método genérico de evaluación (se recomienda no modificar para mantener comparabilidad)
        pass

```

Ajuste de Parámetros de Entrenamiento
En los notebooks de entrenamiento se pueden ajustar constantes asociadas a la temporalidad y al ciclo de entrenamiento:

### Parámetros de ventana temporal

```python
WINDOW = 24
HORIZON = 1

# Entrenamiento del modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,          # Número de épocas
    batch_size=128,     # Tamaño de batch
    class_weight=class_weight,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)
```

## 5. Obsolescencia del Modelo

Cuando la precisión se degrada de forma irreversible debido a cambios estructurales en los datos reales del rancho, se recomienda:

Reajustar hiperparámetros y ventanas temporales en el código base existente.
Si los cambios operativos son sustanciales (e.g., adición de nuevas máquinas o rediseño de flujos de ordeño), descontinuar las versiones actuales e iniciar un nuevo ciclo de proyecto de investigación adaptado a las condiciones actuales del negocio.
