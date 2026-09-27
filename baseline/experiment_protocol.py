"""Shared CPU controls for paired A/B trajectories."""
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_state import stage_seed
from baseline.result_io import write_json


def arm_order(arms, seed, trajectory):
    order = list(arms)
    random.Random(stage_seed(seed, 'order', trajectory)).shuffle(order)
    return order


def record_protocol(output, args, arms, template, hotspots, target, target_hotspots):
    from run_state import sha256
    write_json(Path(output)/'experiment.json', dict(
        config=vars(args), arms=arms, template_sha256=sha256(template), target_sha256=sha256(target),
        binder_hotspots=hotspots, target_hotspots=target_hotspots,
        seed_protocol='sha256-stage-v1', validation_models=['model_1_ptm'],
        arm_order={str(i):arm_order(arms,args.seed,i) for i in range(1,args.n+1)}))
