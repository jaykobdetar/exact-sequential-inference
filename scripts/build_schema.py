"""Regenerate the static JSON Schema (no dependency required)."""
import json
from pathlib import Path


def obj(properties, **extra):
    return {"type":"object", "properties":properties, "required":list(properties), "additionalProperties":False, **extra}


def arr(items, low=0, high=None):
    return {"type":"array", "items":items, "minItems":low, **({"maxItems":high} if high is not None else {})}


def integer(low, high):
    return {"type":"integer", "minimum":low, "maximum":high}


rat={"type":"string", "pattern":r"^(0|[1-9][0-9]*)/[1-9][0-9]*$", "maxLength":4096, "description":"Canonical reduced nonnegative rational. Reduction, probability bounds, and sums are checked by the Python semantic validator."}
ident={"type":"string","pattern":"^[A-Za-z][A-Za-z0-9_\\-]{0,79}$"}
nullable=lambda x:{"anyOf":[x,{"type":"null"}]}
prob2=arr(rat,2,2)
active_input=obj({"hypotheses":{**arr(ident,2,4),"uniqueItems":True},"tests":{**arr(ident,1,4),"uniqueItems":True},"prior":arr(rat,2,4),"p_one":arr(arr(rat,2,4),1,4),"costs":arr(rat,1,4),"loss":arr(arr(rat,2,4),2,4),"initial_horizon":integer(0,3),"history":arr(obj({"test":integer(0,3),"outcome":integer(0,1)}),0,3)})
temporal_input=obj({"entities":{**arr(ident,1,16),"uniqueItems":True},"steps":integer(1,64),"prior":prob2,"transition":arr(prob2,2,2),"sources":arr(obj({"source_id":ident,"emission":arr(prob2,2,2)}),1,4)})
report=obj({"report_id":ident,"origin_report_id":ident,"entity":ident,"event_time":integer(0,63),"arrival_time":integer(0,63),"source_id":ident,"value":integer(0,1)})
query=obj({"query_id":ident,"kind":{"enum":["current_filter","historical_smoother","historical_as_known"]},"entity":ident,"target_time":integer(0,63),"cutoff":integer(0,63)})
actionvalue={"type":"array","prefixItems":[integer(0,7),rat],"items":False,"minItems":2,"maxItems":2}
historytuple={"type":"array","prefixItems":[integer(0,3),integer(0,1)],"items":False,"minItems":2,"maxItems":2}
active_target=obj({"status":{"enum":["ok","impossible_evidence"]},"evidence_probability":rat,"posterior":nullable(arr(rat,2,4)),"action_values":arr(actionvalue,0,8),"minimum_expected_loss":nullable(rat),"selected_action":nullable(integer(0,7))})
node=obj({"history":arr(historytuple,0,3),"remaining":integer(0,3),"mass":rat,"posterior":nullable(arr(rat,2,4)),"values":arr(actionvalue,0,8),"best":nullable(rat),"optimal":{**arr(integer(0,7),0,8),"uniqueItems":True}})
active_trace=obj({"action_order":obj({"guesses":arr(integer(0,3),2,4),"tests":arr(integer(2,7),1,4)}),"root":{"type":"string"},"nodes":arr(node,1,249)})
target=obj({"query_id":ident,"status":{"enum":["ok","impossible_evidence"]},"posterior":nullable(prob2),"global_evidence_probability":rat,"entity_evidence_probability":rat})
entity_summary=obj({"origin_ids":arr(ident,0,8192),"mass":rat})
entity_full=obj({**entity_summary["properties"],"emissions":arr(prob2,1,64),"alpha":arr(prob2,1,64),"beta":arr(prob2,1,64)})
cutoff=obj({"global_mass":rat,"entities":{"type":"object","additionalProperties":{"oneOf":[entity_summary,entity_full]},"minProperties":1,"maxProperties":16}})
temporal_trace=obj({"cutoffs":{"type":"object","patternProperties":{"^(0|[1-9][0-9]?)$":cutoff},"additionalProperties":False,"minProperties":1,"maxProperties":64}})
audit_common={"renderer":{"enum":["standard","reserved_pretty"]},"variant":ident,"label_origin":{"const":"exact_formal_computation"},"independent_oracle":{"type":"string"}}
audit_active=obj({**audit_common,"optimal_actions":arr(integer(0,7),0,8),"prior_axis":integer(0,1),"cost_axis":{"enum":["low","high"]},"loss_axis":{"enum":["symmetric","skew"]},"group_index":integer(0,19)})
audit_anchor=obj({**audit_common,"optimal_actions":arr(integer(0,7),0,8),"anchor":{"const":"balanced_entropy_vs_loss_relevant_test"},"entropy_order_certificate":obj({"denominator_multiplier":integer(1,1000),"exp_negative_scaled_conditional_entropy":arr(rat,1,4),"max_information_tests":arr(integer(0,3),1,4)})})
audit_temporal=obj({**audit_common,"hazard":{"enum":["1/100","2/25"]},"source_error":{"enum":["0/1","1/5"]},"family_index":integer(0,9),"report_schedule":{"const":"one_per_time_entity_t_mod16"},"latent_states_released":{"const":False}})
common={"schema_version":{"const":"exact_sequential_inference/1.0"},"id":ident,"world_group_id":ident,"split":{"enum":["train","validation","test_id","ood_likelihood","ood_surface","anchors"]},"mode":{"enum":["active_sensing","noisy_temporal"]},"formal_input":{},"source_records":{},"queries":{},"prompt":{"type":"object","additionalProperties":{"type":"string"},"minProperties":1,"maxProperties":64},"oracle_targets":{},"exact_trace":{},"paired_variant_id":ident,"audit":{}}
a=obj({**common,"mode":{"const":"active_sensing"},"formal_input":active_input,"source_records":arr({},0,0),"queries":arr({},0,0),"oracle_targets":active_target,"exact_trace":active_trace,"audit":{"oneOf":[audit_active,audit_anchor]}})
b=obj({**common,"mode":{"const":"noisy_temporal"},"formal_input":temporal_input,"source_records":arr(report,0,8192),"queries":arr(query,1,64),"oracle_targets":arr(target,1,64),"exact_trace":temporal_trace,"audit":audit_temporal})
schema={"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:exact-sequential-inference:record:1.0","title":"Exact sequential inference record v1.0","description":"JSON Schema supplies shape checks. Python semantic validation additionally enforces exact rational arithmetic, strict bool/int distinctions, all timing/origin constraints, canonical groups, prompts, exact targets, and traces.","oneOf":[a,b]}
Path("schemas/record-v1.schema.json").write_text(json.dumps(schema,indent=2)+"\n")
config=obj({"schema_version":{"const":"exact_sequential_inference-config/1.0"},"seed":integer(0,2**63-1),"active_groups_per_split":integer(1,20),"temporal_families_per_cell_split":integer(1,10),"active_splits":{**arr({"enum":["train","validation","test_id","ood_likelihood","ood_surface"]},1,5),"uniqueItems":True},"temporal_splits":{**arr({"enum":["train","validation","test_id","ood_surface"]},1,4),"uniqueItems":True},"max_dataset_bytes":integer(1000,20000000),"temporal_schedule":{"const":"one_per_time_entity_t_mod16"}})
config.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:exact-sequential-inference:config:1.0"})
Path("schemas/config-v1.schema.json").write_text(json.dumps(config,indent=2)+"\n")
