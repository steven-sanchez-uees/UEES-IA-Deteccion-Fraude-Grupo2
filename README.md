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

## Conexión con Proyecto Final y Hoja de Ruta de Mejoras

La implementación actual de la red neuronal fue desarrollada desde cero utilizando únicamente NumPy, lo que permitió comprender en detalle cada etapa del proceso: inicialización de pesos con métodos Xavier/He, forward propagation, cálculo de pérdidas, backpropagation manual y optimización mediante gradient descent con ajuste de umbrales para mejorar la detección de fraude. Este enfoque es ideal como prueba de concepto académica y para prototipos en datasets sintéticos o de tamaño reducido, sirviendo como base sólida para el proyecto final.

## Limitaciones y Escalabilidad

Si bien el prototipo es funcional y didáctico, presenta limitaciones para un despliegue productivo:
    * **Eficiencia computacional**: el entrenamiento es secuencial y no aprovecha GPU ni procesamiento paralelo, lo que limita la capacidad de manejar grandes volúmenes de transacciones en tiempo real.
    * **Manejo de datos a gran escala**: la carga completa en memoria no es viable para millones de registros.
    Falta de técnicas avanzadas como early stopping, regularización (Dropout, BatchNorm) o ajuste adaptativo de la tasa de aprendizaje.
    * **Pipeline de datos básico**: apto para datos limpios y estructurados, pero no para escenarios complejos con streaming o fuentes heterogéneas.

## Plan de transición a frameworks avanzados

Para el proyecto final, que trabajará con datos reales de detección de fraude en telecomunicaciones con alto volumen y necesidad de respuesta rápida, se plantea la migración a TensorFlow o PyTorch, lo que permitirá:

    * Aprovechar GPU/TPU para acelerar el entrenamiento.
    * Utilizar autograd para backprop automático y evitar errores manuales.
    * Implementar arquitecturas más complejas (modelos híbridos con embeddings y datos numéricos).
    * Integrar pipelines robustos para el manejo y preprocesamiento eficiente de datos.
    * Aplicar técnicas avanzadas de regularización y optimización.

## Hoja de Ruta para Mejoras Continuas

Este proyecto es una prueba de concepto. Para llevarlo a un entorno de producción, se necesitarían mejoras significativas. El siguiente paso en la hoja de ruta incluye:
* **Migración a Frameworks (TensorFlow/PyTorch)**: Para aprovechar la aceleración por GPU y las optimizaciones de los frameworks modernos.
* **Modelos Avanzados**: Explorar arquitecturas como los `Autoencoders`, que son especialmente potentes para la detección de anomalías y la identificación de patrones de fraude complejos.
* **Modelos Explicables (XAI)**: Implementar librerías como SHAP para generar modelos que no solo detecten el fraude, sino que también expliquen por qué una transacción fue marcada como sospechosa, cumpliendo así con las normativas del sector financiero.
* 

## Componentes reutilizables

De la implementación actual se pueden trasladar directamente:

    * Preprocesamiento de datos (data_preprocessing.py), que estandariza y codifica variables categóricas de forma eficiente.
    * Funciones métricas y cálculo de umbral óptimo para maximizar sensibilidad o F1-score.
    * Estructura modular y configuración de experimentos (experiments.py), que permite comparar arquitecturas y parámetros.
    * Documentación y organización del repositorio, lista para integrarse en un framework más avanzado.

En síntesis, el trabajo desarrollado no solo cumple como ejercicio académico, sino que establece la base técnica y metodológica para evolucionar hacia un sistema escalable, preciso y explicable, capaz de operar en entornos de alta demanda y con datos reales de fraude.
