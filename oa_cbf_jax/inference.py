"""Strict research bundles and inference with explicitly precompiled signatures."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from .dataset import sha256,load_dataset
from .io import write_json
from .models import make_model,GATConfig,unicycle_graph,predict_ensemble
from .uncertainty import cs_disagreement,worst_member_cvar


def export_pilot(members,output):
    if len(members)!=4:raise ValueError('This initial ensemble contract requires four complete members')
    settings=[];states=[];provenance=[]
    for member in map(Path,members):
        if not (member/'complete.json').exists():raise FileNotFoundError(f'Incomplete training run: {member}')
        cfg=json.loads((member/'settings.json').read_text());best=json.loads((member/'best.json').read_text())
        if (Path(cfg['dataset'])/'INVALIDATED.json').exists():raise ValueError('Cannot export a member trained on an invalidated dataset')
        if best['settings_sha256']!=sha256(member/'settings.json'):raise ValueError('Training settings checksum mismatch')
        path=member/'checkpoints'/best['state_file']
        if sha256(path)!=best['state_sha256']:raise ValueError('Member checksum mismatch')
        state=serialization.msgpack_restore(path.read_bytes())
        states.append(state['params']);settings.append(cfg)
        provenance.append(dict(directory=str(member.resolve()),best_epoch=best['epoch'],seed=cfg['seed'],weights_sha256=best['state_sha256']))
    for key in ('flight_warmstart_source','flight_warmstart_source_manifest_sha256','flight_constraint_features_contract','flight_gain_attention_contract','flight_frozen_predictor_contract'):
        if any(s.get(key)!=settings[0].get(key) for s in settings):raise ValueError('Different flight warm-start treatment')
    for key in ['architecture','normalization','dataset_manifest_sha256','targets']:
        if any(s[key]!=settings[0][key] for s in settings):raise ValueError(f'Incompatible member {key}')
    for key in ('local_unicycle_reflection','local_unicycle_expansion','local_unicycle_supervision','local_unicycle_stop_contrast','local_unicycle_prediction_selection','variance_weighted_objective','offline_variance_objective_pilot','variance_refit_contract','variance_refit_source','variance_refit_source_manifest_sha256','variance_refit_mode','risk_distribution_contract','normalized_risk_censor_floor','risk_tail_objective','flight_training_expansion','ensemble_sampling','flight_viable_progress_contract'):
        if any(s.get(key)!=settings[0].get(key) for s in settings):
            raise ValueError(f'Incompatible member {key}')
    if 'ensemble_sampling' in settings[0]:
        from .ensemble_sampling import validate_contract
        for setting in settings:validate_contract(setting)
    domain=settings[0].get('gain_domain',dict(lower=.3,upper=4.))
    gain_dimension=settings[0].get('gain_dimension',2)
    if gain_dimension not in (1,2,4) or any(s.get('gain_dimension',2)!=gain_dimension for s in settings):raise ValueError('Incompatible gain dimension')
    if gain_dimension==4:
        from .quad3d_learning_contract import validate_model
        for setting in settings:validate_model(setting)
    if any(s.get('gain_domain',dict(lower=.3,upper=4.))!=domain for s in settings):raise ValueError('Incompatible member gain domain')
    controller=settings[0].get('controller',dict(sensor_margin_scale=0.))
    if any(s.get('controller',dict(sensor_margin_scale=0.))!=controller for s in settings):raise ValueError('Incompatible member controller')
    if len({s['seed'] for s in settings})!=4:raise ValueError('Duplicate ensemble seeds')
    params=jax.tree.map(lambda *v:np.stack(v),*states)
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(params)):raise ValueError('Nonfinite model weights')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    path=root/'weights.msgpack';path.write_bytes(serialization.msgpack_serialize(params))
    info=dict(schema='oa_cbf_jax_research_bundle_v1',architecture=settings[0]['architecture'],normalization=settings[0]['normalization'],
              targets=settings[0]['targets'],dataset_manifest_sha256=settings[0]['dataset_manifest_sha256'],
              graph_features=settings[0].get('graph_features',18),dataset_schema=settings[0].get('dataset_schema','oa_cbf_initial_state_pilot_v1'),
              events=settings[0].get('events',['collision_first','solver_failure_first']),
              gain_domain=domain,gain_dimension=gain_dimension,
              controller=controller,
              weights_sha256=sha256(path),members=provenance,production_eligible=False,calibration=None,
              limitation='Uncalibrated development ensemble. No closed-loop superiority or deployment safety calibration established.')
    for field in ('nearest_fc_contract','unicycle_constraint_features_contract','training_obstacle_capacity','training_scene_distribution','training_obstacle_count_histogram','bicycle_contract','quad3d_contract','normalization_sampling','progress_contrast_weight','progress_contrast_normalization','progress_contrast_scale_floor','bicycle_constraint_features_contract','bicycle_candidate_encoding_contract','bicycle_motion_history_contract','bicycle_motion_history_source','bicycle_motion_history_source_sha256','bicycle_motion_adapter_contract','bicycle_motion_adapter_source','bicycle_motion_adapter_source_sha256'):
        if field in settings[0]:
            if any(s.get(field)!=settings[0][field] for s in settings):raise ValueError('Ensemble count/distribution provenance mismatch')
            info[field]=settings[0][field]
    for field in ('local_unicycle_contract','local_unicycle_supervision','local_unicycle_expansion','local_unicycle_stop_contrast','local_unicycle_prediction_selection','flight_local_residual_contract','flight_local_residual_source','flight_local_residual_source_manifest_sha256','bicycle_gain_contract','offline_wide_gain_pilot','bicycle_task_progress_contract','variance_weighted_objective','offline_variance_objective_pilot'):
        if field in settings[0]:
            if any(s.get(field)!=settings[0][field] for s in settings):raise ValueError('Ensemble gain contract mismatch')
            info[field]=settings[0][field]
    if info.get('offline_wide_gain_pilot'):
        from .bicycle_gain_contract import model_bank
        model_bank(info)
    for key in ('flight_warmstart_source','flight_warmstart_source_manifest_sha256','flight_constraint_features_contract','flight_gain_attention_contract','flight_frozen_predictor_contract','variance_refit_contract','variance_refit_source','variance_refit_source_manifest_sha256','variance_refit_mode','risk_distribution_contract','normalized_risk_censor_floor','risk_tail_objective','flight_training_expansion','ensemble_sampling','flight_viable_progress_contract'):
        if key in settings[0]:info[key]=settings[0][key]
    if 'clipped_risk_components' in settings[0]:
        if any(s.get('clipped_risk_components')!=settings[0]['clipped_risk_components'] for s in settings):
            raise ValueError('Incompatible component count')
        info['clipped_risk_components']=settings[0]['clipped_risk_components']
    write_json(root/'manifest.json',info)
    if 'local_unicycle_reflection' in settings[0]:
        from .local_unicycle_reflection_training import bind_export
        bind_export(members,root)
    return root


class ResearchPredictor:
    def __init__(self,bundle,allow_uncalibrated=False,device=None):
        root=Path(bundle);self.metadata=json.loads((root/'manifest.json').read_text())
        if 'risk_distribution_contract' in self.metadata:
            raise ValueError('Clipped-risk development bundle requires dedicated evaluation; ordinary Gaussian calibration/policy cannot load it')
        if (root/'INVALIDATED.json').exists():raise ValueError('Invalidated inference bundle; inspect INVALIDATED.json')
        if self.metadata['schema']!='oa_cbf_jax_research_bundle_v1':raise ValueError('Unknown inference schema')
        if not allow_uncalibrated:raise ValueError('Research bundle requires explicit uncalibrated diagnostic mode; not an operational OA-CBF policy')
        if len(self.metadata['members'])!=4:raise ValueError('Missing ensemble members')
        self.gain_dimension=self.metadata.get('gain_dimension',2)
        if self.gain_dimension not in (1,2,4):raise ValueError('Unsupported gain dimension')
        if self.gain_dimension==4:
            from .quad3d_learning_contract import validate_model
            validate_model(self.metadata)
        path=root/'weights.msgpack'
        if sha256(path)!=self.metadata['weights_sha256']:raise ValueError('Inference checksum mismatch')
        raw=serialization.msgpack_restore(path.read_bytes())
        self.device=device or jax.devices()[0]
        self.model=make_model(GATConfig(**self.metadata['architecture']))
        if self.model.config.flight_gain_attention:
            from .quad2d_gain_attention import contract as gain_attention_contract
            if self.metadata.get('flight_gain_attention_contract')!=gain_attention_contract() or self.metadata.get('graph_features')!=40:
                raise ValueError('Missing candidate-obstacle attention provenance')
        if self.model.config.flight_constraint_features:
            from .quad2d_constraint_features import contract
            if (self.metadata.get('flight_constraint_features_contract') != contract()
                    or self.metadata.get('graph_features') != 40):
                raise ValueError('Missing observed flight coefficient provenance')
        if self.model.config.encoder == 'nearest_fc':
            from .nearest_fc import validate_metadata
            validate_metadata(self.metadata)
        readout_refit='event_readout_refit' in self.metadata
        if readout_refit:
            from .bicycle_event_bundle import validate
            validate(root,self.metadata)
        if self.model.config.compute_dtype=='float64':
            if not readout_refit:
                from .precision_bundle import validate_derivation
                validate_derivation(root,self.metadata)
            # Explicit FP64 arithmetic, with separate lineage validation for
            # unchanged-weight ports and newly fitted adverse readouts.
            raw=jax.tree.map(lambda v:jnp.asarray(v,dtype=jnp.float64),raw)
        self.params=jax.device_put(raw,self.device)
        norm=self.metadata['normalization']
        mean=jax.device_put(np.asarray(norm['target_mean'],np.float32),self.device)
        scale=jax.device_put(np.asarray(norm['target_scale'],np.float32),self.device)
        def predict_features(params,features,node_mask,gains):
            out=predict_ensemble(self.model,params,features,node_mask,gains)
            mu=out['mean']*scale+mean;variance=jnp.exp(out['log_variance'])*scale**2
            # [E,B,K,D] -> [B,K,E,D] for analytic disagreement.
            risk_means=jnp.transpose(mu[...,:1],(1,2,0,3));risk_variances=jnp.transpose(variance[...,:1],(1,2,0,3))
            return dict(mean=mu,variance=variance,event_probability=jax.nn.sigmoid(out['event_logits']),
                        disagreement=cs_disagreement(risk_means,risk_variances),
                        finite_member_cvar=worst_member_cvar(risk_means[...,0],risk_variances[...,0]))
        def predict(params,x,goal,obstacles,mask,gains):
            features,node_mask=jax.vmap(unicycle_graph)(x,goal,obstacles,mask)
            return predict_features(params,features,node_mask,gains)
        self._function=jax.jit(predict);self._compiled={}
        self._feature_function=jax.jit(predict_features);self._compiled_features={}

    def warm(self,batch=1,capacity=16,candidates=64):
        if self.metadata.get('graph_features',18)!=18:raise ValueError('Use warm_features for a route-context bundle')
        key=(batch,capacity,candidates)
        with jax.default_device(self.device):
            args=(self.params,jnp.zeros((batch,4),jnp.float32),jnp.ones((batch,2),jnp.float32),
                  jnp.zeros((batch,capacity,5),jnp.float32),jnp.zeros((batch,capacity),bool),jnp.ones((batch,candidates,2),jnp.float32))
            start=time.perf_counter();executable=self._function.lower(*args).compile()
            jax.block_until_ready(executable(*args));elapsed=time.perf_counter()-start
        self._compiled[key]=executable
        return elapsed

    def predict(self,x,goal,obstacles,mask,gains):
        key=(x.shape[0],obstacles.shape[1],gains.shape[1])
        if key not in self._compiled:raise ValueError(f'Unwarmed signature {key}; runtime compilation is forbidden')
        # Host staging explicitly avoids unsafe peer copies between GPUs.
        args=tuple(jax.device_put(np.asarray(v),self.device) for v in (x,goal,obstacles,mask,gains))
        return self._compiled[key](self.params,*args)

    def warm_features(self,batch=1,nodes=18,candidates=64):
        features=self.metadata.get('graph_features',18);key=(batch,nodes,candidates,features)
        with jax.default_device(self.device):
            args=(self.params,jnp.zeros((batch,nodes,features),jnp.float32),jnp.ones((batch,nodes),bool),
                  jnp.ones((batch,candidates,self.gain_dimension),jnp.float32))
            start=time.perf_counter();executable=self._feature_function.lower(*args).compile()
            jax.block_until_ready(executable(*args));elapsed=time.perf_counter()-start
        self._compiled_features[key]=executable
        return elapsed

    def predict_features(self,features,mask,gains):
        key=(features.shape[0],features.shape[1],gains.shape[1],features.shape[2])
        if key not in self._compiled_features:raise ValueError(f'Unwarmed feature signature {key}; runtime compilation is forbidden')
        if mask.shape!=features.shape[:2] or gains.shape!=(features.shape[0],gains.shape[1],self.gain_dimension):raise ValueError('Invalid feature inference shape')
        args=tuple(jax.device_put(np.asarray(a,dtype=bool if i==1 else np.float32),self.device) for i,a in enumerate((features,mask,gains)))
        return self._compiled_features[key](self.params,*args)


def evaluate_pilot(bundle,dataset,output):
    predictor=ResearchPredictor(bundle,allow_uncalibrated=True)
    data=load_dataset(dataset,'validation')
    if sha256(Path(dataset)/'manifest.json')!=predictor.metadata['dataset_manifest_sha256']:raise ValueError('Wrong evaluation dataset')
    n,capacity,k=len(data['group_id']),data['obstacles'].shape[1],data['gains'].shape[1]
    predictor.warm_features(n,data['features'].shape[1],k)
    predictions=jax.tree.map(np.asarray,predictor.predict_features(data['features'],data['node_mask'],data['gains']))
    mean=predictions['mean'].mean(axis=0);probability=predictions['event_probability'].mean(axis=0)
    mask=data['target_mask'];target=data['target']
    norm=predictor.metadata['normalization']
    brier=np.sum(np.where(data['event_mask'],(probability-data['events'])**2,0.),axis=(0,1))/np.maximum(np.sum(data['event_mask'],axis=(0,1)),1)
    metrics=dict(validation_groups=n,continuous={},first_event_brier=brier.tolist())
    for head,name in enumerate(predictor.metadata['targets']):
        valid=mask[...,head];actual=target[...,head][valid];pred=mean[...,head][valid]
        metrics['continuous'][name]=dict(rmse=float(np.sqrt(np.mean((actual-pred)**2))),
                    train_mean_reference_rmse=float(np.sqrt(np.mean((actual-norm['target_mean'][head])**2))),count=int(valid.sum()))
    # Conditional ranking diagnostic only: all candidates must have observed
    # progress so an unknown/censored label never masquerades as slow progress.
    valid_group=np.all(mask[...,1],axis=1)
    actual=target[valid_group,:,1];predicted=mean[valid_group,:,1]
    replicas=json.loads((Path(dataset)/'manifest.json').read_text()).get('replicas',1)
    if replicas>1:
        actual=actual.reshape(len(actual),k//replicas,replicas).mean(axis=-1)
        predicted=predicted.reshape(len(predicted),k//replicas,replicas).mean(axis=-1)
    chosen=np.argmax(predicted,axis=1)
    regret=np.max(actual,axis=1)-actual[np.arange(len(actual)),chosen]
    metrics['conditional_candidate_ranking']=dict(groups=int(valid_group.sum()),mean_regret=float(np.mean(regret)) if len(regret) else None,
             query_slot_mean_regrets=np.mean(np.max(actual,axis=1)[:,None]-actual,axis=0).tolist() if len(actual) else [],
             limitation='Conditions on all candidates/replicas having observed progress; replicas averaged before ranking. Query slots may vary across scenes; not a closed-loop completion result')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    lineage={key:data[key] for key in ('group_id','query_tick') if key in data}
    np.savez_compressed(out/'validation_predictions.npz',**lineage,**predictions)
    write_json(out/'metrics.json',metrics);print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export-pilot');e.add_argument('--members',nargs=4,required=True);e.add_argument('--output',required=True)
    v=sub.add_parser('evaluate-pilot');v.add_argument('--bundle',required=True);v.add_argument('--dataset',required=True);v.add_argument('--output',required=True)
    args=vars(p.parse_args());command=args.pop('command')
    export_pilot(**args) if command=='export-pilot' else evaluate_pilot(**args)
