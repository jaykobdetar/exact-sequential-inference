import copy
import importlib.util
import json
from pathlib import Path
from functools import lru_cache
import tempfile
import unittest

from exact_sequential_inference import active, temporal
from exact_sequential_inference.common import InvalidInput, dumps, load_json
from exact_sequential_inference.generate import active_records, temporal_records, config_check, record
from exact_sequential_inference.grouping import temporal_group
from exact_sequential_inference.validate import validate_record


@lru_cache(None)
def _release_fixture(mode, split):
    config=load_json('configs/smoke.json')
    if mode=='active_sensing':
        config['active_splits']=['train' if split=='anchors' else split]
        return next(r for r in active_records(config) if r['split']==split)
    config['temporal_splits']=[split]
    return next(r for r in temporal_records(config) if r['audit']['source_error']=='1/5' and r['audit']['variant']=='base')


def release_fixture(mode,split='train'):
    return copy.deepcopy(_release_fixture(mode,split))


class ContractTests(unittest.TestCase):
    def active_record(self):
        return release_fixture('active_sensing')

    def temporal_record(self):
        return release_fixture('noisy_temporal')

    def test_records_and_strict_target_trace_mutations(self):
        a=self.active_record()
        self.assertTrue(validate_record(a))
        mutations=[lambda r:r['oracle_targets'].__setitem__('selected_action',False),lambda r:r['audit']['optimal_actions'].__setitem__(0,False),lambda r:r['exact_trace']['nodes'][0].__setitem__('remaining',True),lambda r:r['audit'].__setitem__('group_index',True),lambda r:r.__setitem__('world_group_id','active_wrong'),lambda r:r.__setitem__('paired_variant_id','pair_wrong'),lambda r:r['prompt'].__setitem__('decision',r['prompt']['decision']+' target=0'),lambda r:r.__setitem__('extra',1),lambda r:r['oracle_targets'].__setitem__('posterior',['2/4','1/2'])]
        for i,mutation in enumerate(mutations):
            r=copy.deepcopy(a);mutation(r)
            with self.subTest(mutation=i),self.assertRaises(InvalidInput):
                validate_record(r)
        t=self.temporal_record()
        self.assertTrue(validate_record(t))
        view=json.loads(t['prompt']['q0_historical_as_known_7_7'])
        self.assertTrue(all(x['arrival_time']<=7 for x in view['source_records']))
        for mutation in [lambda r:r['oracle_targets'][0].__setitem__('posterior',[True,'0/1']),lambda r:r['audit'].__setitem__('latent_states_released',0),lambda r:r['prompt'].__setitem__('query',dumps({'source_records':r['source_records']})),lambda r:next(iter(r['exact_trace']['cutoffs'].values())).__setitem__('global_mass','2/2')]:
            r=copy.deepcopy(t);mutation(r)
            with self.assertRaises(InvalidInput):validate_record(r)

    def test_all_renderer_record_roundtrips(self):
        for baseline in (release_fixture('active_sensing','ood_surface'),release_fixture('noisy_temporal','ood_surface')):
            for renderer in ("standard","reserved_pretty"):
                audit={k:v for k,v in baseline['audit'].items() if k not in {'renderer','variant','label_origin','independent_oracle','optimal_actions'}}
                variant=f"{audit['cost_axis']}_{audit['loss_axis']}_{renderer}" if baseline['mode']=='active_sensing' else ('reserved_renderer' if renderer=='reserved_pretty' else 'base')
                r=record(baseline['mode'],baseline['world_group_id'],baseline['split'],baseline['formal_input'],baseline['source_records'],baseline['queries'],renderer,variant,baseline['paired_variant_id'],audit)
                self.assertTrue(validate_record(json.loads(dumps(r))))

    def test_optional_json_schema(self):
        if importlib.util.find_spec('jsonschema') is None:
            self.skipTest('optional jsonschema not installed; strict standard-library validator remains required')
        import jsonschema
        schema=load_json('schemas/record-v1.schema.json')
        validator=jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        for r in (self.active_record(),self.temporal_record()):
            validator.validate(r)
        bad=self.active_record();bad['formal_input']['initial_horizon']=True
        with self.assertRaises(jsonschema.ValidationError):validator.validate(bad)

    def test_config_and_json_duplicate_keys(self):
        c=load_json('configs/smoke.json')
        self.assertEqual(c,config_check(c))
        for key,value in [('seed',True),('active_groups_per_split',False),('temporal_families_per_cell_split',0),('active_splits',['train','train']),('temporal_schedule','dense'),('max_dataset_bytes',30000000)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(InvalidInput):config_check(bad)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for text in ('{"x":1,"x":2}','{"x":NaN}'):
                p.write_text(text)
                with self.assertRaises(InvalidInput):load_json(p)


if __name__=='__main__':unittest.main()
