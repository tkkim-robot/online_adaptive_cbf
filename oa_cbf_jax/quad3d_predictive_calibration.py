"""Reserved-parent conditional prediction calibration for genuine Quad3D bundles."""
import argparse
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from .quad3d_learning_contract import validate_training_dataset,validate_model,read
from .quad3d_candidate_data import candidate_bank,REPLICAS
from .dataset import load_dataset,sha256
from .inference import ResearchPredictor
from .models import predict_ensemble
from .bicycle_predictive_calibration import fit_parameters,diagnostics
from .metrics import parent_mean
from .io import write_json

SCHEMA='oa_cbf_quad3d_reserved_predictive_v95'


def calibrate(bundle,dataset,output,batch=32,runtime_qualification=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);dataset=Path(dataset)
    m=validate_training_dataset(dataset);model=ResearchPredictor(bundle,allow_uncalibrated=True)
    if model.model.config.encoder=='nearest_fc':
        from .nearest_fc_qualification import validate_qualification
        validate_qualification(runtime_qualification,bundle,dataset)
    validate_model(model.metadata,m)
    if model.model.config.compute_dtype!='float32':raise ValueError('This contract uses original trained FP32 neural inference')
    if sha256(dataset/'manifest.json')!=model.metadata['dataset_manifest_sha256']:raise ValueError('Wrong weight-training lineage')
    bank=candidate_bank(m['schema']).astype(np.float32);mean=jnp.asarray(model.metadata['normalization']['target_mean'],jnp.float32);scale=jnp.asarray(model.metadata['normalization']['target_scale'],jnp.float32)
    def raw(params,features,mask,gains):
        out=predict_ensemble(model.model,params,features,mask,gains)
        return dict(mean=out['mean']*scale+mean,variance=jnp.exp(out['log_variance'])*scale**2,event_logits=out['event_logits'])
    nodes=m['capacity']+2
    fn=jax.jit(raw);args=(model.params,jnp.zeros((batch,nodes,m['graph_features']),jnp.float32),jnp.ones((batch,nodes),bool),jnp.asarray(np.broadcast_to(bank,(batch,16,4))))
    start=time.perf_counter();exe=fn.lower(*args).compile();jax.block_until_ready(exe(*args));cold=time.perf_counter()-start
    def predict(data):
        result=[]
        for start in range(0,len(data['group_id']),batch):
            n=min(batch,len(data['group_id'])-start)
            features=np.pad(data['features'][start:start+n],((0,batch-n),(0,0),(0,0)))
            mask=np.pad(data['node_mask'][start:start+n],((0,batch-n),(0,0)))
            p=jax.device_get(exe(model.params,jnp.asarray(features),jnp.asarray(mask),args[-1]))
            result.append({k:np.repeat(v[:,:n],REPLICAS,axis=2) for k,v in p.items()})
        return {k:np.concatenate([v[k] for v in result],axis=1) for k in result[0]}
    reservation={r['group_id']:r['partition'] for r in read(Path(m['source'])/'parents.json')}
    def subset(role):
        d=load_dataset(dataset,role)
        if set(d['group_id'])!={k for k,v in reservation.items() if v==role}:raise ValueError('Missing reserved physical parent')
        np.testing.assert_array_equal(d['gains'],np.broadcast_to(np.repeat(bank,REPLICAS,axis=0),d['gains'].shape))
        return d
    fit_data=subset('prediction_fit');fit_prediction=predict(fit_data)
    parameters=fit_parameters(fit_prediction,fit_data)
    fit=dict(schema=SCHEMA,stage='reserved_prediction_fit_only',bundle=str(Path(bundle).resolve()),
        weights_sha256=model.metadata['weights_sha256'],bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
        dataset=str(dataset.resolve()),dataset_manifest_sha256=sha256(dataset/'manifest.json'),dataset_index_sha256=sha256(dataset/'index.json'),
        dataset_audit_sha256=sha256(dataset/'independent_replay.json'),quad3d_contract=model.metadata['quad3d_contract'],
        controller=m['controller'],targets=m['targets'],events=m['events'],gain_domain=m['gain_domain'],candidates=bank.tolist(),
        group_ids={role:sorted(k for k,v in reservation.items() if v==role) for role in ('prediction_fit','prediction_audit')},
        **parameters,prediction_batch=batch,compiled_prediction_signatures=1,compile_seconds=cold,
        event_budget_statistic='Maximum-member calibrated any_adverse_termination. Conditional collision-first head excluded without positive support.',
        trajectory_gate_ready=False,production_eligible=False,whole_goal_complete=False,
        limitation='Development conditional prediction adjustment on declared acquired histories, not an adaptive-trajectory, OOD, rare-collision, posterior or physical CVaR guarantee. Fresh frozen-policy trajectory gating and closed-loop audits remain required.')
    # The dataset validator checks integrity of every split. No audit outcome
    # enters fitting; freeze fit before audit prediction/diagnostic evaluation.
    if model.model.config.encoder=='nearest_fc':
        fit.update(nearest_fc_contract=model.metadata['nearest_fc_contract'],
            runtime_qualification=str(Path(runtime_qualification).resolve()),runtime_qualification_sha256=sha256(runtime_qualification))
    write_json(root/'prediction_fit.json',fit);fit_sha=sha256(root/'prediction_fit.json')
    audit_data=subset('prediction_audit');audit_prediction=predict(audit_data)
    validation=subset('validation');validation_prediction=predict(validation)
    mu=validation_prediction['mean'].mean(0);ids=validation['group_id']
    actual=validation['target'][...,1].reshape(len(ids),16,REPLICAS).mean(-1)
    predicted=mu[...,1].reshape(len(ids),16,REPLICAS).mean(-1)
    chosen=predicted.argmax(-1)
    validation_metrics=dict(parents=len(set(ids)),queries=len(ids),
        parent_rmse=[float(np.sqrt(parent_mean((mu[...,h]-validation['target'][...,h])**2,ids,validation['target_mask'][...,h]))) for h in range(2)],
        train_mean_reference_rmse=[float(np.sqrt(parent_mean((model.metadata['normalization']['target_mean'][h]-validation['target'][...,h])**2,ids,validation['target_mask'][...,h]))) for h in range(2)],
        offline_mean_progress_regret=parent_mean(actual.max(-1)-actual[np.arange(len(ids)),chosen],ids),
        limitation='Offline model diagnostics only; no analytic gain search is deployed or promoted as a comparator.')
    report=dict(prediction_fit_sha256=fit_sha,fit=diagnostics(fit_prediction,fit_data,fit),audit=diagnostics(audit_prediction,audit_data,fit),
        validation=validation_metrics,model_promoted=False,whole_goal_complete=False)
    for role,d,pred in [('prediction_fit',fit_data,fit_prediction),('prediction_audit',audit_data,audit_prediction),('validation',validation,validation_prediction)]:
        np.savez_compressed(root/(role+'_predictions.npz'),group_id=d['group_id'],query_tick=d['query_tick'],**pred)
    assert fn._cache_size()==0
    write_json(root/'prediction_audit.json',report)
    write_json(root/'complete.json',dict(status='completed',prediction_fit_sha256=fit_sha,prediction_audit_sha256=sha256(root/'prediction_audit.json'),
        implicit_jit_cache_entries=0,predictions={role:sha256(root/(role+'_predictions.npz')) for role in ('prediction_fit','prediction_audit','validation')},whole_goal_complete=False))
    print(json.dumps(dict(stage='prediction_calibrated',parameters=parameters,diagnostics=report)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bundle','dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--runtime-qualification');calibrate(**vars(p.parse_args()))
