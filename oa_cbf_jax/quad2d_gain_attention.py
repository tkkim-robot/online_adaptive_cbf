"""Gain-conditioned pooling of observed graph nodes, with no controller search."""


def contract():
    return dict(schema='quad2d_candidate_obstacle_attention_v1',
        scope='Static graph40 Gaussian GAT with original pooled history-invariant scene encoder.',
        input='Observed graph nodes and candidate gains only; no rollouts, labels or latent state.',
        motivation='A candidate-independent scene summary can discard which obstacle limits a particular gain pair.',
        computation='Encode the graph once, then multihead candidate-query attention over its obstacle nodes; residual into the original ego context.',
        initialization='Zero output projection preserves the corresponding original GAT member exactly at epoch zero.',
        empty_obstacles='Residual identically zero; masked nodes never receive attention.',
        controller_changed=False,labels_changed=False,gain_domain_changed=False)


def frozen_predictor_contract():
    return dict(schema='quad2d_gain_attention_frozen_predictor_v1',
        trainable=['gain_query','obstacle_key','gain_context_residual'],
        frozen='Every original graph encoder, pooled context and prediction-head parameter.',
        optimizer='Mask gradients before clipping; mask AdamW updates including weight decay.',
        initialization='Original corresponding ensemble member, new output projection zero, epoch zero eligible.',
        reason='Separate learning the added attention from destabilizing the already trained predictor.',
        labels_changed=False,controller_changed=False,inference_changed=False)


def trainable_attention_mask(params):
    import jax
    import numpy as np
    names=set(frozen_predictor_contract()['trainable'])
    if not names.issubset(params):raise ValueError('Missing candidate attention parameters')
    return {key:jax.tree.map(lambda value:np.full(value.shape,key in names,bool),subtree)
            for key,subtree in params.items()}
