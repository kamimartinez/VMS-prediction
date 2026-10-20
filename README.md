# VMS-prediction

Proyecto de análisis de datos enfocado en comprender y predecir la formación de colas en el Sistema de Ordeño Voluntario (VMS) del CAETEC.

**Objetivo general:** identificar patrones temporales y condiciones recurrentes asociados con la formación de colas en el sistema de ordeño automático, con el fin de generar información que contribuya a comprender su funcionamiento y apoyar la toma de decisiones.

**Objetivos específicos:**
- **Segmentar periodos de operación:** aplicar técnicas de *clustering* para distinguir periodos con y sin formación de colas según el nivel de ocupación del robot.
- **Identificar patrones recurrentes:** utilizar reglas de asociación para encontrar combinaciones de variables temporales y operativas relacionadas con la formación de colas.
- **Predecir la formación de colas:** desarrollar un modelo de clasificación que estime la presencia de colas en periodos específicos a partir de variables temporales y operativas.

## Metodología

El proyecto sigue la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining) de forma iterativa. Las fases se trabajan en función de los hallazgos y las necesidades del proyecto, por lo que es posible regresar a etapas anteriores para revisar objetivos, validar supuestos o ajustar el análisis a los datos disponibles.

La documentación se organiza según las iteraciones realizadas y las fases de CRISP-DM involucradas. Los números de las carpetas identifican la secuencia de los entregables documentados, mientras que los números entre paréntesis corresponden a las fases de CRISP-DM y a sus respectivas versiones.

1. Entendimiento del negocio (1.0)
2. Entendimiento de los datos (2.0)
3. Entendimiento del negocio (1.1)
4. Entendimiento de los datos (2.1)
5. Preparación de los datos (3.0)

Esta secuencia representa el trabajo documentado hasta el momento. Las siguientes actividades se incorporarán conforme avance el proyecto, incluyendo el modelado, la evaluación y, si corresponde, el despliegue.

## Estructura del repositorio

La estructura actual del repositorio incluye los siguientes elementos:

```text
VMS-prediction/
├── crisp-dm/
│   ├── 01-business-understanding-1.0/
│   ├── 02-data-understanding-2.0/
│   ├── 03-business-understanding-1.1/
│   ├── 04-data-understanding-2.1/
│   └── 05-data-preparation-3.0/
├── docs/
│   └── data-privacy/
├── notebooks/
│   ├── data/
│   ├── modeling/
│   └── evaluation/
├── reports/
│   ├── figures/
│   └── tables/
├── .gitignore
├── LICENSE
└── README.md
```

Se utilizan archivos .gitkeep para conservar en Git los directorios que todavía no contienen entregables.

- **`crisp-dm/`**: contiene la documentación de las iteraciones y fases de la metodología CRISP-DM desarrolladas hasta el momento.
- **`docs/`**: agrupa documentación complementaria al proceso de minería de datos.
  - **`data-privacy/`**: contiene los documentos relacionados con la privacidad y la seguridad de los datos elaborados por los integrantes del equipo.
- **`notebooks/`**: contiene los notebooks utilizados para el desarrollo técnico del proyecto.
  - **`data/`**: incluye los notebooks de entendimiento y preparación de los datos.
  - **`modeling/`**: espacio destinado a los notebooks de modelado.
  - **`evaluation/`**: espacio destinado a los notebooks relacionados con la evaluación de los modelos creados.
- **`reports/`**: espacio destinado a los resultados y entregables gráficos del proyecto.
  - **`figures/`**: destinado a las figuras y visualizaciones.
  - **`tables/`**: destinado a las tablas y resultados tabulares.
- **`.gitignore`**: define los archivos y patrones que Git debe ignorar.
- **`LICENSE`**: contiene la licencia del repositorio.
- **`README.md`**: presenta el propósito, la metodología y la estructura del proyecto.

La estructura podrá ampliarse conforme avance el proyecto y se generen nuevos entregables.

## Estado y seguimiento del proyecto

En este espacio se incorporará el **Plan de Valor Ganado (PVG)** del proyecto, con el propósito de dar seguimiento al avance de las actividades y comparar el progreso real con la planificación establecida.

- **Plan de Valor Ganado:** [Consultar PVG](https://docs.google.com/spreadsheets/d/18ZSEoYqNdcYhLaWvrS6nrfhg-XO51X0N6ywaMQQgBhQ/edit?usp=sharing).

Los avances, resultados y cambios en la planificación se actualizarán conforme se desarrollen las actividades del proyecto.

## Documentación

La documentación metodológica se encuentra organizada dentro de `crisp-dm/`, siguiendo las iteraciones realizadas.

Para conocer el contexto, los objetivos de negocio, los criterios de éxito y los objetivos de minería de datos, se recomienda comenzar por los siguientes documentos:

- `crisp-dm/XX-business-understanding-[Ultima versión]/business-understanding.pdf`: documento de entendimiento del negocio.
- `crisp-dm/XX-business-understanding-[Ultima versión]/one-page-summary.pdf`: resumen ejecutivo del entendimiento del negocio.

Estos documentos corresponden a la versión inicial del entendimiento del negocio y podrán actualizarse conforme avance el proyecto.
