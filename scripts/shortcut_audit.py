"""Fixed untrained diagnostics. They never supply generator targets."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exact_sequential_inference.common import dumps,qstr
from exact_sequential_inference.temporal import available_origins

counts=Counter();matches=Counter();squared=defaultdict(float)
active_by_pair=defaultdict(set)
cadences=[]
for line in (Path(sys.argv[1])/'records.jsonl').open():
    r=json.loads(line)
    if r['mode']=='active_sensing':
        target=r['oracle_targets']
        if target['status']!='ok':continue
        counts['active']+=1
        opts=r['audit']['optimal_actions']
        matches['always_guess0']+=int(0 in opts)
        matches['always_first_legal_test']+=int(next((a for a,v in target['action_values'] if a>=4),-1) in opts)
        active_by_pair[r['paired_variant_id']].add(target['selected_action'])
    else:
        if r['audit']['variant']=='base':
            origins=available_origins(r['source_records'],63)
            cadence=Counter(x['entity'] for x in origins)
            assert len(origins)==64 and set(cadence.values())=={4}
            assert sorted(x['event_time'] for x in origins)==list(range(64))
            assert all(x['entity']==r['formal_input']['entities'][x['event_time']%16] for x in origins)
            cadences.append({'family':r['world_group_id'],'origins':64,'reports_per_entity':4})
        targets={t['query_id']:t for t in r['oracle_targets']}
        for q in r['queries']:
            target=targets[q['query_id']]
            if target['status']!='ok':continue
            counts['temporal_queries']+=1
            observed=[x for x in available_origins(r['source_records'],q['cutoff']) if x['entity']==q['entity']]
            truth=Fraction(target['posterior'][1])
            latest=max(observed,key=lambda x:(x['event_time'],x['origin_report_id']))['value'] if observed else Fraction(1,2)
            mean=Fraction(sum(x['value'] for x in observed),len(observed)) if observed else Fraction(1,2)
            weighted=[(Fraction(1,2)**(q['cutoff']-x['event_time']),x['value']) for x in observed]
            decay=sum(w*v for w,v in weighted)/sum(w for w,v in weighted) if weighted else Fraction(1,2)
            for name,value in [('latest_report',Fraction(latest)),('report_fraction',mean),('fixed_half_life_one',decay),('declared_prior',Fraction(r['formal_input']['prior'][1]))]:
                squared[name]+=float((value-truth)**2)
                matches[name]+=int(value==truth)
print(dumps({'counts':dict(counts),'exact_matches':dict(matches),'temporal_mean_squared_error_p1_float_summary':{k:v/counts['temporal_queries'] for k,v in squared.items()},'cost_loss_pairs_with_changed_canonical_action':sum(len(s)>1 for s in active_by_pair.values()),'primary_temporal_cadence_checks':len(cadences),'primary_temporal_cadence_passed':True,'limitations':'Fixed rules only, no fitting. Duplicates/order variants are correlated, so scores are descriptive, not independent-sample estimates. Identifier/template correlations are not evidence of learned robustness.'}))
