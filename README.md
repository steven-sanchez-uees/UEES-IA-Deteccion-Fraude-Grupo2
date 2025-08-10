# Sistema de Detección de Fraude en Transacciones Financieras

## Resumen

Este proyecto aborda un problema crítico en el sector de las fintech: la detección de fraude en transacciones digitales. El desafío principal es doble: por un lado, minimizar las pérdidas millonarias causadas por transacciones fraudulentas y, por otro, evitar los falsos positivos que degradan la experiencia del usuario. Un modelo que marca como sospechosa una transacción legítima puede generar frustración y hacer que los clientes dejen de usar el servicio. Por esta razón, el proyecto busca un equilibrio entre la detección efectiva del fraude y la minimización de las interrupciones para los usuarios legítimos.

La solución que exploramos aquí es la implementación desde cero de una **Red Neuronal Artificial (RNA)**. Este enfoque nos ha permitido un entendimiento profundo y práctico de los mecanismos internos de estos modelos, como la inicialización de pesos, las funciones de activación y los procesos de propagación hacia adelante y atrás.

## Estructura del Proyecto
La estructura del repositorio está diseñada para ser modular y escalable, facilitando la reproducibilidad de los experimentos y la organización de los componentes.

````
├── data/
│   └── datos_sinteticos_fraude.csv
│
├── docs/
│   └── reporte_tecnico.md
│
├── notebooks/
│   ├── 01_implementacion_red.ipynb
│   │   # Implementación desde cero de la RNA y prueba con el problema XOR.
│   ├── 02_aplicacion_fraude.ipynb
│   │   # Adaptación de la RNA al problema de detección de fraude.
│   ├── 03_experimentacion_comparativa.ipynb
│   │   # Comparación del rendimiento de la RNA con modelos de referencia.
│   └── 04_subir_github.ipynb
│       # Script para gestionar el repositorio.
│
├── results/
│   ├── f1_comparison.png
│   ├── matriz_confusion_optima.png
│   ├── metrics_optimal_threshold.csv
│   ├── performance_comparison.csv
│   ├── training_curve_*.png
│   └── training_curves_xor.png
│
└── src/
    ├── data_preprocessing.py
    |   # Lógica para preprocesar y estandarizar los datos.
    ├── experiments.py
    |   # Script para orquestar la ejecución de múltiples experimentos.
    └──neural_network.py
        # Implementación manual de la red neuronal.
````

## Metodología y Resultados

La metodología del proyecto se centra en un flujo de trabajo claro:
1. **Implementación de Componentes**: Creación de los módulos principales de la red neuronal y preprocesamiento de datos.
2. **Generación de Datos**: Simulación de un conjunto de datos desequilibrado para el problema de fraude.
3. **Entrenamiento y Evaluación**: Entrenamiento del modelo con los datos sintéticos, con un enfoque en la optimización del umbral de clasificación para maximizar métricas como el `F1-Score`.
4. **Análisis Comparativo**: Se compararon diferentes arquitecturas de red neuronal y variaciones de hiperparámetros contra una **línea de base** de Regresión Logística.

Los resultados detallados de estos experimentos, incluyendo las métricas de rendimiento y las matrices de confusión, se encuentran en el `informe_tecnico.md` y en la carpeta `results/`. El análisis final de estos resultados es crucial para determinar la viabilidad y las limitaciones de nuestra implementación.

## Hoja de Ruta para Mejoras Continuas

Este proyecto es una prueba de concepto. Para llevarlo a un entorno de producción, se necesitarían mejoras significativas. El siguiente paso en la hoja de ruta incluye:
* **Migración a Frameworks (TensorFlow/PyTorch)**: Para aprovechar la aceleración por GPU y las optimizaciones de los frameworks modernos.
* **Modelos Avanzados**: Explorar arquitecturas como los `Autoencoders`, que son especialmente potentes para la detección de anomalías y la identificación de patrones de fraude complejos.
* **Modelos Explicables (XAI)**: Implementar librerías como SHAP para generar modelos que no solo detecten el fraude, sino que también expliquen por qué una transacción fue marcada como sospechosa, cumpliendo así con las normativas del sector financiero.
