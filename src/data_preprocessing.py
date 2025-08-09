
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple

def preprocess_data(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, StandardScaler, OneHotEncoder]:
    """
    Preprocesa un DataFrame para ser usado en la red neuronal.
    Aplica:
    - Estandarización a columnas numéricas.
    - One-Hot Encoding a columnas categóricas.
    - Combina las características en una matriz de NumPy.
    - Separa las etiquetas (fraude).
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada no puede estar vacío.")

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Preprocesar columnas numéricas
    X_num = scaler.fit_transform(df[num_cols])
    
    # Preprocesar columnas categóricas
    X_cat = encoder.fit_transform(df[cat_cols])
    
    # Combinar todas las características
    X = np.hstack([X_num, X_cat])
    y = df['fraude'].values.reshape(-1, 1)
    
    return X, y, scaler, encoder
