"""Versioned compact AF2 artifacts; preserve arrays exactly, never quantize."""
import numpy as np

COMPACT_FIELDS = ('pae', 'atom_positions', 'atom_mask', 'aatype', 'residue_index',
                  'plddt', 'ptm', 'i_ptm')


def prediction_artifact(aux, mode='full'):
    if mode == 'full':
        return aux
    if mode != 'compact':
        raise ValueError('Unknown prediction artifact mode')
    if 'pae' not in aux or 'atom_positions' not in aux:
        raise ValueError('Compact predictions require PAE and unrounded coordinates')
    result = {key:np.asarray(aux[key]) for key in COMPACT_FIELDS if key in aux}
    result['_artifact_schema'] = 'foldcraft-prediction-compact-v1'
    return result
