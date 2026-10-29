"""Negative controls for scientifically meaningful release metadata.

Formal counterfactuals are re-solved and re-identified before validation, so
rejection cannot depend on a stale ID, numerical target, trace or prompt.
"""
import copy
from pathlib import Path
import unittest
from exact_sequential_inference import active
from exact_sequential_inference.common import InvalidInput, digest, load_json
from exact_sequential_inference.generate import record
from exact_sequential_inference.grouping import temporal_group
from exact_sequential_inference.release_contract import ReleaseCoverage
from exact_sequential_inference.validate import validate_record
from test_contract import release_fixture


def rebuild(r):
    audit={k:v for k,v in r['audit'].items() if k not in {'renderer','variant','label_origin','independent_oracle','optimal_actions'}}
    if r['mode']=='active_sensing':
        if r['split']=='anchors':
            from exact_sequential_inference.entropy_anchor import certificate
            audit['entropy_order_certificate']=certificate(r['formal_input'])
        group=active.group_id(r['formal_input']['p_one'])
        pair='pair_'+digest([group,r['formal_input']['initial_horizon']] if r['split']=='anchors' else [group,r['formal_input']['prior'],r['formal_input']['history']])
    else:
        group=temporal_group(r['formal_input'],r['source_records'])
        pair='pair_'+group
    return record(r['mode'],group,r['split'],r['formal_input'],r['source_records'],r['queries'],r['audit']['renderer'],r['audit']['variant'],pair,audit)


class ReleaseBindingTests(unittest.TestCase):
    def reject_mutations(self, baseline, mutations):
        for label,mutate in mutations:
            row=copy.deepcopy(baseline)
            mutate(row)
            row=rebuild(row)
            with self.subTest(label=label),self.assertRaises(InvalidInput):
                validate_record(row)

    def test_reviewer_counterexamples(self):
        directory=Path('reports/history/reviewer-g2-v1.0.1')
        if not directory.is_dir():
            self.skipTest('historical reviewer fixtures are not bundled; generated counterexamples run separately')
        for name in ('ordinary_likelihood_record_relabelled_ood','wrong_temporal_hazard_axis','wrong_active_prior_axis'):
            with self.subTest(name=name),self.assertRaises(InvalidInput):
                validate_record(load_json(directory/(name+'.json')))

    def test_active_axes_grid_variant_and_names(self):
        baseline=release_fixture('active_sensing')
        self.assertTrue(validate_record(baseline))
        mutations=[
            ('ordinary labelled OOD',lambda r:r.__setitem__('split','ood_likelihood')),
            ('prior axis',lambda r:r['audit'].__setitem__('prior_axis',1)),
            ('cost axis',lambda r:r['audit'].__setitem__('cost_axis','high')),
            ('loss axis',lambda r:r['audit'].__setitem__('loss_axis','skew')),
            ('formal prior',lambda r:r['formal_input'].__setitem__('prior',['1/10','1/5','3/10','2/5'])),
            ('one formal cost',lambda r:r['formal_input']['costs'].__setitem__(0,'1/50')),
            ('one formal loss',lambda r:r['formal_input']['loss'][0].__setitem__(1,'2/1')),
            ('off-grid likelihood',lambda r:r['formal_input']['p_one'][0].__setitem__(0,'1/2')),
            ('wrong namespace',lambda r:r['formal_input']['hypotheses'].__setitem__(0,'reservedH0')),
            ('reserved renderer in train',lambda r:r['audit'].__setitem__('renderer','reserved_pretty')),
            ('variant disagrees',lambda r:r['audit'].__setitem__('variant','high_symmetric_standard')),
            ('horizon outside primary support',lambda r:r['formal_input'].__setitem__('initial_horizon',2)),
            ('history outside primary support',lambda r:r['formal_input'].__setitem__('history',[{'test':1,'outcome':0}]))]
        self.reject_mutations(baseline,mutations)
        self.reject_mutations(release_fixture('active_sensing','ood_likelihood'),[
            ('OOD labelled ID',lambda r:r.__setitem__('split','test_id'))])
        self.reject_mutations(release_fixture('active_sensing','ood_surface'),[
            ('surface labelled ID',lambda r:r.__setitem__('split','test_id'))])

    def test_valid_active_axis_counterfactuals(self):
        row=release_fixture('active_sensing')
        row['audit']['prior_axis']=1
        row['formal_input']['prior']=['1/10','1/5','3/10','2/5']
        row['audit']['cost_axis']='high'
        row['formal_input']['costs']=['3/1']*4
        row['audit']['loss_axis']='skew'
        row['formal_input']['loss']=[[str(0 if a==h else (5 if h==0 else 1))+'/1' for h in range(4)] for a in range(4)]
        row['audit']['variant']='high_skew_standard'
        self.assertTrue(validate_record(rebuild(row)))

    def test_temporal_axes_cadence_copies_and_regime(self):
        row=release_fixture('noisy_temporal')
        self.assertTrue(validate_record(row))
        mutations=[
            ('hazard axis',lambda r:r['audit'].__setitem__('hazard','2/25')),
            ('error axis',lambda r:r['audit'].__setitem__('source_error','0/1')),
            ('wrong transition',lambda r:r['formal_input'].__setitem__('transition',[['23/25','2/25'],['2/25','23/25']])),
            ('wrong source emission',lambda r:r['formal_input']['sources'][1].__setitem__('emission',[['1/1','0/1'],['0/1','1/1']])),
            ('wrong prior',lambda r:r['formal_input'].__setitem__('prior',['1/3','2/3'])),
            ('unsupported temporal OOD',lambda r:r.__setitem__('split','ood_likelihood')),
            ('base labelled duplicates',lambda r:r['audit'].__setitem__('variant','duplicates')),
            ('wrong record order',lambda r:r['source_records'].reverse()),
            ('missing original',lambda r:r['source_records'].pop()),
            ('query omitted',lambda r:r['queries'].pop()),
            ('reserved renderer in train',lambda r:r['audit'].__setitem__('renderer','reserved_pretty')),
            ('invalid primary delay',lambda r:r['source_records'][0].__setitem__('arrival_time',r['source_records'][0]['event_time']+2))]
        self.reject_mutations(row,mutations)
        extra=copy.deepcopy(row['source_records'][0]);extra['report_id']='extra_'+extra['origin_report_id']
        self.reject_mutations(row,[('extra copy labelled base',lambda r:r['source_records'].append(extra))])
        self.reject_mutations(release_fixture('noisy_temporal','ood_surface'),[
            ('reserved namespace relabelled ID',lambda r:r.__setitem__('split','test_id'))])

    def test_valid_temporal_axis_counterfactual(self):
        row=release_fixture('noisy_temporal')
        row['audit']['hazard']='2/25'
        row['formal_input']['transition']=[['23/25','2/25'],['2/25','23/25']]
        row['audit']['source_error']='0/1'
        for source in row['formal_input']['sources']:
            source['emission']=[['1/1','0/1'],['0/1','1/1']]
        self.assertTrue(validate_record(rebuild(row)))

    def test_anchor_profile_bound(self):
        self.reject_mutations(release_fixture('active_sensing','anchors'),[
            ('anchor table',lambda r:r['formal_input']['p_one'][0].__setitem__(0,'1/5')),
            ('anchor cost',lambda r:r['formal_input'].__setitem__('costs',['0/1']*4))])

    def test_configuration_slots_and_completeness(self):
        config=load_json('configs/smoke.json')
        for mode,index_field in [('active_sensing','group_index'),('noisy_temporal','family_index')]:
            row=release_fixture(mode)
            row['audit'][index_field]=1
            # A provenance index in range for the format is outside this config.
            with self.assertRaises(InvalidInput):ReleaseCoverage(config).add(row)
        coverage=ReleaseCoverage(config)
        row=release_fixture('active_sensing')
        coverage.add(row)
        coverage.add(row)
        with self.assertRaises(InvalidInput):coverage.finish()
        different=copy.deepcopy(row)
        different['world_group_id']='active_'+'f'*64
        with self.assertRaises(InvalidInput):coverage.add(different)
        config['active_splits']=['test_id']
        with self.assertRaises(InvalidInput):ReleaseCoverage(config).add(row)


if __name__=='__main__':unittest.main()
