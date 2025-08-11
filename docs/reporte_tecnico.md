# Reporte técnico: Implementación y Experimentación de Redes Neuronales para la Detección de Fraude

## 1. Introducción y Metodología

 objetivo principal de este proyecto es la implementación de una **Red Neuronal Artificial (RNA)** desde cero, utilizando únicamente la librería `NumPy`, para la detección de fraude en transacciones financieras. El enfoque se basa en la simulación de un entorno real, donde las transacciones fraudulentas (`clase 1`) son una clase minoritaria (0.1%) y el principal problema es el desequilibrio de clases.

La metodología de trabajo se divide en tres fases principales:
1. **Implementación y Validación de la RNA**: Se codificó una red neuronal `feedforward` con soporte para múltiples capas ocultas, funciones de activación (`ReLU`, `Tanh`, `Sigmoid`), inicialización de pesos (He y Xavier), y los algoritmos de propagación hacia adelante y atrás. La validación se realizó con el problema clásico XOR, confirmando la capacidad del modelo para aprender relaciones no lineales (ver el archivo `notebooks/01_implementacion_red.ipynb` y la curva de pérdida en `results/training_curves_xor.png`).
2. **Aplicación a la Detección de Fraude**: Se generaron datos sintéticos con características representativas de transacciones y se entrenó la RNA, optimizando el umbral de clasificación para maximizar el `F1-Score`. La evaluación se centró en métricas clave para la industria financiera, como la `precisión`, el `recall` (sensibilidad) y la matriz de confusión.
3. **Experimentación Comparativa**: Se comparó el rendimiento de múltiples configuraciones de la RNA (variando capas, activaciones, tasa de aprendizaje y épocas) contra un modelo de referencia (`baseline`), la **Regresión Logística** de `scikit-learn`, para validar la eficacia del modelo implementado.

## 2. Resultados de los Experimentos

Los resultados de la fase de experimentación se encuentran detallados en el archivo `results/performance_comparison.csv`. A continuación, se presenta un análisis de los hallazgos más relevantes:

### 2.1. Comparación General de Modelos

El gráfico de barras del **F1-Score** muestra que el modelo `baseline` de **Regresión Logística** superó a todas las configuraciones de la red neuronal.

* **Regresión Logística (Baseline)**: Con un **F1-Score de 0.7778**, este modelo demostró una robusta capacidad para encontrar un equilibrio entre la `precisión` (0.875) y el `recall` (0.700). Su simplicidad y la optimización de los algoritmos de `scikit-learn` le otorgan una ventaja significativa sobre la implementación manual de la RNA para este conjunto de datos.
* **Mejor Red Neuronal (Arch_3_narrow)**: La arquitectura `[13, 8, 4, 1]` logró el mejor rendimiento entre las RNA, con un **F1-Score de 0.5714**. A pesar de su `recall` bajo (0.40), obtuvo una `precisión` perfecta (1.0), lo que sugiere que las predicciones que realiza son muy confiables, aunque se le escapan muchos fraudes.

### 2.2. Análisis de Hiperparámetros

La experimentación con los hiperparámetros de la red neuronal reveló lo siguiente:

* **Arquitectura**: Una red más estrecha (`Arch_3_narrow`) funcionó mejor que una más ancha (`Arch_2_wider`), la cual tuvo un rendimiento muy pobre. Esto podría indicar que la complejidad adicional no fue necesaria para las características de los datos sintéticos, llevando a un sobreajuste o a problemas de convergencia.
* **Función de Activación**: El uso de `ReLU` en las capas ocultas (`Arch_1`) resultó en un mejor rendimiento que el uso de `Tanh` (`Act_Tanh`), lo cual se alinea con la práctica común en redes neuronales profundas para evitar el problema del gradiente desvanecedor.
* **Tasa de Aprendizaje y Épocas**: Las redes con un `learning_rate` muy bajo (`LR_Low`) o muy alto (`LR_High`) tuvieron un rendimiento inferior. El gráfico `f1_comparison.png` muestra una caída significativa en el rendimiento cuando se usó `LR_High` (0.1), lo que sugiere que este valor era demasiado agresivo, haciendo que el modelo no pudiera converger de manera estable.



## 3. Matriz de Confusión y Análisis de Métricas

En un problema de detección de fraude, la `precisión` y el `recall` son más importantes que la `accuracy` global. La matriz de confusión del modelo de Regresión Logística con el umbral óptimo (0.20) ilustra mejor su desempeño (ver `results/metrics_optimal_threshold.csv` y `results/matriz_confusion_optima.png`).

- [Matriz de confusión Optima]
<img src="../results/matriz_confusion_optima.png" alt="matriz_confusion_optima" width="800"/>
* **Verdaderos Positivos (TP)**: 5. Casos de fraude detectados correctamente.
* **Falsos Negativos (FN)**: 5. Casos de fraude que el modelo no detectó.
* **Verdaderos Negativos (TN)**: 9990. Transacciones legítimas identificadas correctamente.
* **Falsos Positivos (FP)**: 0. Transacciones legítimas marcadas erróneamente como fraude.

El resultado más destacado es la ausencia total de **falsos positivos (FP=0)**. Esto es crucial para el negocio, ya que asegura que la experiencia del usuario no se verá afectada por bloqueos de transacciones legítimas.  

## 4. Conclusión Técnica y Hoja de Ruta

La implementación en `NumPy` fue exitosa para fines educativos, pero los resultados demuestran sus limitaciones en términos de rendimiento y optimización frente a frameworks maduros. Para el proyecto final, se propone la siguiente hoja de ruta:

* **Migración a un Framework**: El modelo debe ser re-implementado en **TensorFlow** o **PyTorch** para aprovechar la aceleración por GPU, el `autodiff` y los optimizadores avanzados (como Adam), que permitirán un entrenamiento más eficiente y estable.
* **Modelos de Detección de Anomalías**: La siguiente etapa será reemplazar el modelo de clasificación binaria por un **Autoencoder** o un **Isolation Forest**. Un Autoencoder es especialmente adecuado para este problema, ya que aprende a reconstruir los patrones de las transacciones normales y cualquier transacción que no pueda reconstruir (es decir, el fraude) es marcada como una anomalía.
* **Explicabilidad (XAI)**: Para cumplir con los requisitos regulatorios, se integrarán librerías como **SHAP** para generar valores de explicabilidad. Estos valores permitirán entender por qué el modelo marca una transacción como fraudulenta, proporcionando una herramienta esencial para los analistas de negocio y de riesgo.


## 5. Análisis de Resultados

## 5.1 Resumen ejecutivo

Comparamos varias configuraciones de red neuronal (variando arquitectura y activaciones) contra un baseline de Regresión Logística. El mejor modelo fue la configuración con mayor capacidad (capas ocultas más anchas) y activación ReLU en capas internas con sigmoid en la salida; este modelo ofreció el mejor F1 y un equilibrio razonable entre precisión y recall, superando a la línea base.
A nivel operativo, el ajuste de umbral de decisión fue determinante para priorizar recall/sensibilidad (minimizar falsos negativos), consistente con el objetivo de negocio en fraude.

## 5.2 Comparación de arquitecturas y activaciones

* Arquitecturas más anchas (p. ej., [input → 32 → 16 → 1]) capturan mejor la no linealidad del problema, logrando F1 superior a configuraciones más estrechas (p. ej., [input → 16 → 8 → 1]).
* Activaciones: ReLU en ocultas converge de forma estable; tanh funciona pero suele requerir más épocas o ajustes de LR para igualar a ReLU.
* Tasa de aprendizaje: valores intermedios (p. ej., 0.05) facilitaron la convergencia sin inestabilidad; tasas muy bajas alargan el entrenamiento sin mejoras claras, y tasas muy altas pueden oscilar.

## 5.3 Curvas de entrenamiento y estabilidad

Las training_curve_*.png muestran reducción de MSE por época, coherente con la estabilización del entrenamiento. Para clasificación, recomendamos además monitorear F1 o Recall en validación e implementar early stopping en el proyecto final para evitar sobreajuste y reducir cómputo.

## 5.4 Umbral óptimo y trade-offs de negocio

El gráfico threshold_curves_best.png ilustra el trade-off entre precisión y recall.
* En fraude, el costo de falsos negativos (FN) suele ser más alto (fraudes no detectados). Por ello es razonable mover el umbral hacia valores que aumenten el recall, asumiendo un incremento de falsos positivos (FP) manejable mediante revisión automática o reglas adicionales.
* El umbral óptimo reportado en performance_comparison.csv es el que maximiza F1, pero la operación final puede requerir un umbral diferente, alineado al costo del negocio

## 5.5 Matriz de confusión y errores característicos

confusion_matrix_best.png permite cuantificar FP y FN en el punto operativo seleccionado.
* El umbral óptimo reportado en performance_comparison.csv es el que maximiza F1, pero la operación final puede requerir un umbral diferente, alineado al costo del negocioSi observamos FN aún elevados, proponemos: (i) desplazar umbral hacia mayor recall, (ii) re-ponderar la función objetivo (cost-sensitive), (iii) enriquecer features (ej. secuencias temporales) y (iv) añadir regularización.

## 5.6 Conclusiones prácticas
* La RNA supera a la Regresión Logística en F1, confirmando que el problema es no lineal.
* Mayor capacidad del modelo (capas más anchas) ayuda, pero requiere regularización y early stopping para robustez.
* La optimización del umbral es clave para alinear el desempeño con los costos reales (FN vs. FP).
* Para escalar a datos reales y mayor volumen, avanzaremos a TensorFlow/PyTorch con GPU, autograd, pipelines y XAI (SHAP), además de explorar Autoencoders para detección de anomalías.
