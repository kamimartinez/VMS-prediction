# VMS-prediction

La descripción y objetivos del proyecto se pueden encontrar en la sección de **Business Understanding** del directorio de ``crisp-dm/``.

## Metodología

El proyecto sigue la metodología **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), que organiza el trabajo en seis fases:

1. Entendimiento del negocio (*Business Understanding*).
2. Entendimiento de los datos (*Data Understanding*).
3. Preparación de los datos (*Data Preparation*).
4. Modelado (*Modeling*).
5. Evaluación (*Evaluation*).
6. Despliegue (*Deployment*).

La documentación del proyecto se organiza tomando como referencia estas fases.

## Estructura del repositorio

La estructura actual del repositorio incluye los siguientes elementos:

```text
VMS-prediction/
├── crisp-dm/
│   ├── 01-business-understanding/
│   ├── 02-data-understanding/
│   ├── 03-data-preparation/
│   ├── 04-modeling/
│   ├── 05-evaluation/
│   └── 06-deployment/
├── Privacidad de los Datos/
├── docs/
├── notebooks/
│   ├── data/
│   ├── deployment/
│   └── modeling/
├── reports/
│   ├── figures/
│   └── tables/
├── .gitignore
├── LICENSE
└── README.md
```

- **`crisp-dm/`**: contiene la documentación correspondiente a las fases de la metodología.
- **`Privacidad de los Datos/`**: agrupa documentación relacionada con la privacidad y seguridad de la información (se incluirá en un futuro dentro de docs/).
- **`docs/`**: contiene aquella documentación externa a las fases de la metodología.
- **`notebooks/`**: notebooks dedicados a los tres principales aspectos de la metología en cuanto a código: data, modeling y deployment.
- **`.gitignore`**: define archivos y patrones que Git debe ignorar.
- **`LICENSE`**: contiene la licencia del repositorio.
- **`README.md`**: contiene detalles de la metodología y la estructura del repositorio.

La estructura podrá ampliarse conforme avance el proyecto y se generen nuevos entregables.

## Estado del proyecto

El proyecto cuenta con documentación de *Business Understanding*, que establece el contexto del negocio, los objetivos, los criterios de éxito, los riesgos y la planificación inicial.

Las actividades posteriores de entendimiento y preparación de los datos, modelado y evaluación deberán documentarse conforme se desarrollen. Los resultados y el cumplimiento de los criterios de aceptación se determinarán durante el análisis; no se presuponen resultados anticipados.

## Documentación

La documentación de las fases de CRISP-DM se encuentra organizada dentro del directorio `crisp-dm/`. Para comprender el contexto y los objetivos del proyecto, se recomienda comenzar por los entregables de `01-business-understanding/`.