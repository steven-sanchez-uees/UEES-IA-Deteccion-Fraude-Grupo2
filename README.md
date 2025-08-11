# Sistema de Detección de Fraude en Transacciones Financieras

## Resumen

Este proyecto aborda un desafío clave para el sector fintech: detectar fraudes en transacciones digitales minimizando el impacto sobre clientes legítimos.
El objetivo es lograr un equilibrio entre la máxima detección de fraudes y la mínima generación de falsos positivos, ya que cada operación legítima bloqueada deteriora la experiencia del usuario y puede generar pérdidas de clientes.

La solución desarrollada es un prototipo de Red Neuronal Artificial (RNA) implementada íntegramente en NumPy, lo que permitió un control total sobre el flujo de datos, inicialización de pesos, funciones de activación y procesos de entrenamiento. Este enfoque garantiza un entendimiento profundo de la lógica interna del modelo y sienta las bases para una transición a entornos productivos.

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
│   ├── training_curve_LR_Low.png
│   ├── training_curve_LR_High.png
│   ├── training_curve_Epochs_High.png
│   ├── training_curve_Arch_3_narrow.png
│   ├── training_curve_Arch_2_wider.png
│   ├── training_curve_Arch_1.png
│   ├── training_curve_Act_Tanh.png
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

## Hallazgos Claves
* El baseline de Regresión Logística superó a las configuraciones actuales de la RNA en F1-Score, gracias a su simplicidad y optimización interna.
* La mejor RNA alcanzó precisión perfecta (1.0) pero con recall limitado, detectando menos fraudes de los deseados.
* La optimización del umbral de decisión se confirmó como un factor crítico para balancear costos de falsos negativos y falsos positivos.
* Arquitecturas más anchas y ReLU en capas ocultas mejoran la capacidad de detección, pero requieren regularización y early stopping para evitar sobreajuste.

## Limitaciones y Escalabilidad

Si bien el prototipo es funcional y didáctico, presenta limitaciones para un despliegue productivo:
*  **Eficiencia computacional**: el entrenamiento es secuencial y no aprovecha GPU ni procesamiento paralelo, lo que limita la capacidad de manejar grandes volúmenes de transacciones en tiempo real.
* **Manejo de datos a gran escala**: la carga completa en memoria no es viable para millones de registros.
    Falta de técnicas avanzadas como early stopping, regularización (Dropout, BatchNorm) o ajuste adaptativo de la tasa de aprendizaje.
* **Pipeline de datos básico**: apto para datos limpios y estructurados, pero no para escenarios complejos con streaming o fuentes heterogéneas.

## Plan de transición a frameworks avanzados

Para el proyecto final, que trabajará con datos reales de detección de fraude en telecomunicaciones con alto volumen y necesidad de respuesta rápida, se plantea la migración a TensorFlow o PyTorch, lo que permitirá:
1. **Migración a TensorFlow/PyTorch** para aprovechar GPU, autograd y optimizadores avanzados.
2. **Arquitecturas avanzadas** como Autoencoders o modelos híbridos para detección de anomalías complejas.
3. **Integración de XAI (Explainable AI)** con librerías como SHAP para justificar predicciones y cumplir requisitos regulatorios.
4. **Pipelines robustos** para ingesta, limpieza y preprocesamiento de datos en tiempo real.
5. **Regularización y validación avanzada** para mejorar generalización y estabilidad.

## Hoja de Ruta para Mejoras Continuas

La implementación actual de la red neuronal fue desarrollada desde cero utilizando únicamente NumPy, lo que permitió comprender en detalle cada etapa del proceso: inicialización de pesos con métodos Xavier/He, forward propagation, cálculo de pérdidas, backpropagation manual y optimización mediante gradient descent con ajuste de umbrales para mejorar la detección de fraude. Este enfoque es ideal como prueba de concepto académica y para prototipos en datasets sintéticos o de tamaño reducido, sirviendo como base sólida para el proyecto final.

Este proyecto es una prueba de concepto. Para llevarlo a un entorno de producción, se necesitarían mejoras significativas. El siguiente paso en la hoja de ruta incluye:
* **Migración a Frameworks (TensorFlow/PyTorch)**: Para aprovechar la aceleración por GPU y las optimizaciones de los frameworks modernos.
* **Modelos Avanzados**: Explorar arquitecturas como los `Autoencoders`, que son especialmente potentes para la detección de anomalías y la identificación de patrones de fraude complejos.
* **Modelos Explicables (XAI)**: Implementar librerías como SHAP para generar modelos que no solo detecten el fraude, sino que también expliquen por qué una transacción fue marcada como sospechosa, cumpliendo así con las normativas del sector financiero.

## Componentes reutilizables

De la implementación actual se pueden trasladar directamente:

* Preprocesamiento de datos (data_preprocessing.py), que estandariza y codifica variables categóricas de forma eficiente.
* Funciones métricas y cálculo de umbral óptimo para maximizar sensibilidad o F1-score.
* Estructura modular y configuración de experimentos (experiments.py), que permite comparar arquitecturas y parámetros.
* Documentación y organización del repositorio, lista para integrarse en un framework más avanzado.

En síntesis, el trabajo desarrollado no solo cumple como ejercicio académico, sino que establece la base técnica y metodológica para evolucionar hacia un sistema escalable, preciso y explicable, capaz de operar en entornos de alta demanda y con datos reales de fraude.
