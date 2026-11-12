"""Validación por bloques espaciales: las celdas vecinas se parecen, así que se reparten bloques enteros."""

import numpy as np
import pandas as pd

from src.config import BLOCK_SIZE_CELLS, N_FOLDS, RANDOM_STATE

# El identificador de bloque es fila_de_bloque * BLOCK_ROW_FACTOR + columna_de_bloque: único mientras haya menos de
# BLOCK_ROW_FACTOR columnas de bloques (el área piloto tiene ~31)
BLOCK_ROW_FACTOR = 1000


def block_ids(row: pd.Series, col: pd.Series) -> pd.Series:
    """Bloque de BLOCK_SIZE_CELLS x BLOCK_SIZE_CELLS celdas al que pertenece cada celda."""
    return (row // BLOCK_SIZE_CELLS) * BLOCK_ROW_FACTOR + col // BLOCK_SIZE_CELLS


def assign_folds(blocks: pd.Series, flooded: pd.Series) -> pd.Series:
    """Grupo (0 .. N_FOLDS-1) de cada celda según su bloque.

    Los bloques con alguna celda inundada y los secos se barajan y reparten por separado,
    para que todos los grupos tengan las dos clases.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    wet = flooded.groupby(blocks).max().astype(bool)
    fold_of = {}
    for group in (wet[wet].index, wet[~wet].index):
        for i, block in enumerate(rng.permutation(np.asarray(group))):
            fold_of[block] = i % N_FOLDS
    return blocks.map(fold_of)
