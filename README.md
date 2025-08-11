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

## Metodología y Alcance

La metodología del proyecto se centra en un flujo de trabajo claro:
1. Implementación de la RNA desde cero, incluyendo módulos de preprocesamiento, inicialización de pesos (He/Xavier), forward/backpropagation y optimización con ajuste de umbral para maximizar el `F1-Score`.
2. Simulación de datos de fraude con fuerte desbalance de clases para replicar un escenario real.
3. Entrenamiento y evaluación comparando diversas arquitecturas y parámetros frente a un modelo baseline de Regresión Logística.
4. Análisis de resultados con métricas críticas (precisión, recall, F1) y matrices de confusión.

Los resultados detallados y gráficas comparativas se encuentran en `docs/reporte_tecnico.md` y en la carpeta `results/`.

## Hallazgos Claves
* El baseline de Regresión Logística superó a las configuraciones actuales de la RNA en `F1-Score`, gracias a su simplicidad y optimización interna.
* La mejor RNA alcanzó precisión perfecta `(1.0)` pero con recall limitado, detectando menos fraudes de los deseados.
* La optimización del umbral de decisión se confirmó como un factor crítico para balancear costos de falsos negativos y falsos positivos.
* Arquitecturas más anchas y `ReLU` en capas ocultas mejoran la capacidad de detección, pero requieren regularización y early stopping para evitar sobreajuste.

## Limitaciones y Escalabilidad

Si bien el prototipo es funcional y didáctico, presenta limitaciones para un despliegue productivo:
* **Eficiencia:** entrenamiento secuencial sin uso de GPU, poco viable para grandes volúmenes o tiempo real.
* **Escalabilidad:** carga completa en memoria, no apto para millones de registros.
* **Pipeline:** diseñado para datos limpios y estructurados, sin soporte para streaming o fuentes heterogéneas.
* **Optimización:** sin técnicas avanzadas como batch normalization, dropout o ajuste adaptativo de la tasa de aprendizaje.

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

* Preprocesamiento modular (`data_preprocessing.py`): estandarización y codificación eficiente de variables.
* Cálculo de umbral óptimo para maximizar sensibilidad o F1-Score según objetivos de negocio.
* Framework de experimentos (`experiments.py`): permite comparar arquitecturas y parámetros con facilidad.
* Documentación estructurada lista para integrarse en un desarrollo más avanzado.

## Conclusión

El prototipo desarrollado cumple su objetivo como prueba de concepto, permitiendo implementar desde cero una red neuronal artificial aplicada a la detección de fraude y comprendiendo en profundidad sus fundamentos técnicos. La experimentación evidenció fortalezas —como la alta precisión— y limitaciones en recall y escalabilidad, señalando la necesidad de optimización y migración a frameworks como TensorFlow o PyTorch para entornos productivos.

El análisis comparativo con un baseline de Regresión Logística confirmó que, si bien la RNA es competitiva, requiere mejoras para operar con grandes volúmenes de datos y en tiempo real. El proyecto deja como legado componentes modulares reutilizables y un marco metodológico validado, constituyendo una base sólida para evolucionar hacia un sistema de detección de fraude escalable, preciso y explicable, alineado con los objetivos del negocio y las exigencias regulatorias.
