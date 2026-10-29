"""Read-only, source-hashed 2/3-state horizon-zero task-8 compatibility check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from fractions import Fraction
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task8-root', type=Path, help='Optional trusted checkout containing probabilistic_calibration')
args = parser.parse_args()
if args.task8_root is None:
    print(json.dumps({'status': 'skipped', 'reason': 'optional task-8 checkout not supplied'}))
    raise SystemExit(0)
root = args.task8_root.resolve() / 'probabilistic_calibration'
if not all((root / name).is_file() for name in ('common.py', 'solver.py', 'oracle.py')):
    parser.error('--task8-root must contain probabilistic_calibration/{common,solver,oracle}.py')
sys.path.insert(0, str(root.parent))
from exact_sequential_inference.active import solve
from exact_sequential_inference.common import qstr
from probabilistic_calibration.solver import solve as task8_solve
from probabilistic_calibration.oracle import target as task8_oracle

cases=0
for n in (2,3):
    for cells in product((0,1,2),repeat=n):
        for bit in (0,1):
            loss=[[qstr(int(a!=h)) for h in range(n)] for a in range(n)]
            inp={'hypotheses':[f'H{x}' for x in range(n)],'tests':['T0'],'prior':[qstr(Fraction(1,n))]*n,'p_one':[[qstr(Fraction(x,2)) for x in cells]],'costs':['0/1'],'loss':loss,'initial_horizon':1,'history':[{'test':0,'outcome':bit}]}
            adapted={'states':inp['hypotheses'],'prior_counts':[1]*n,'model':{'kind':'sensors','channels':[{'name':'T0','symbols':['zero','one'],'counts':[[2-x,x] for x in cells]}]},'observations':[{'channel':0,'symbol':bit}],'loss':{'predict':loss,'abstain':['100/1']*n}}
            answer,_,optimal=solve(inp)
            legacy=task8_solve(adapted)
            legacy2=task8_oracle(adapted)
            for ref in (legacy,legacy2):
                assert answer['status']==ref['status']
                assert answer['posterior']==ref['posterior']
                assert answer['evidence_probability']==ref['evidence_probability']
                if answer['status']=='ok':
                    assert [v for a,v in answer['action_values']]==ref['expected_losses'][:n]
                    assert optimal==[a for a in ref['optimal_actions'] if a<n]
            cases+=1
print(json.dumps({'cases':cases,'states':[2,3],'task8_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'common.py',root/'solver.py',root/'oracle.py']},'passed':True},sort_keys=True))
