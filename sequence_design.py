"""CPU contracts for ProteinMPNN redesign selections."""
from numbers import Integral
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa


def mutable_positions(binder_len, interface, method):
    if binder_len < 1:
        raise ValueError('Binder must contain residues')
    interface = set(interface)
    if any(not isinstance(r, Integral) or r < 1 or r > binder_len for r in interface):
        raise ValueError('Interface residue outside the binder')
    if method not in ('full', 'non-interface'):
        raise ValueError('Unknown redesign method')
    return [i for i in range(1, binder_len + 1) if method == 'full' or i not in interface]


def redesign(mpnn, pdb, binder_len, interface, method, temperature, samples):
    mutable = mutable_positions(binder_len, interface, method)
    if not mutable:
        structure = PDBParser(QUIET=True).get_structure('designed', pdb)
        model = structure[0]
        seqs = [''.join(seq1(r.resname, custom_map={'MSE':'M'}) for r in model[c] if is_aa(r)) for c in ('A','B')]
        if len(seqs[1]) != binder_len:
            raise ValueError('Designed binder length does not match MPNN contract')
        # One unique unchanged candidate: avoid invented duplicates or target redesign.
        return {'seq': ['/'.join(seqs)], 'redesign_skipped': 'empty mutable set'}
    mpnn.prep_inputs(pdb_filename=pdb, chain='A,B',
                     fix_pos=','.join(f'B{i}' for i in mutable), rm_aa='C', inverse=True)
    return mpnn.sample_parallel(temperature=temperature, batch=samples)
