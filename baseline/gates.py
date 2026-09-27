"""Shared AF2 acceptance rules, including explicit ensemble decisions."""
import pandas as pd


def validation_status(frame):
    if 'validation_pass' not in frame:
        return pd.Series(True, index=frame.index, dtype='boolean')
    # Missing or malformed ensemble decisions remain unknown, never truthy strings.
    return frame['validation_pass'].map(
        lambda v: True if str(v).lower() in ('true','1','1.0') else
        False if str(v).lower() in ('false','0','0.0') else pd.NA).astype('boolean')


def af2_mask(frame):
    p, i, e = (('plddt','iptm','ipae') if 'iptm' in frame else
               ('af2_plddt','af2_iptm','af2_ipae'))
    return ((frame[p] > .8) & (frame[i] > .5) & (frame[e] < .35) &
            validation_status(frame)).fillna(False)
