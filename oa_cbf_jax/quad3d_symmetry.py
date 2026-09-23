"""Exact quarter-turn symmetries of the shared linearized Quad3D task.

The motor permutation also reverses yaw for odd quarter turns. This is an
algebraic symmetry of this linearized plant, not a nonlinear rigid-body claim.
No altered trajectory labels are invented; positions, attitudes, motor forces,
observations and causal bias history transform together. Use only with the
checked square motor layout and symmetric horizontal physical limits.
"""
import jax
import jax.numpy as jnp


def validate(config):
    if config.robot.inertia_x!=config.robot.inertia_y:
        raise ValueError('Quarter-turn augmentation requires equal horizontal inertias')
    if config.nominal_bias_observer!='innovation_ema_v97':
        raise ValueError('Quarter-turn features require the58-column history contract')


def rotate_xy(values,turns):
    matrices=jnp.asarray([[[1,0],[0,1]],[[0,-1],[1,0]],
                          [[-1,0],[0,-1]],[[0,1],[-1,0]]],values.dtype)
    return values@matrices[turns%4].T


def rotate_control(control,turns):
    return jnp.take(control,(jnp.arange(4)-turns)%4,axis=-1)


def rotate_state(state,turns):
    output=state.at[...,0:2].set(rotate_xy(state[...,0:2],turns))
    output=output.at[...,6:8].set(rotate_xy(state[...,6:8],turns))
    # Horizontal acceleration=[g*pitch,-g*roll].
    output=output.at[...,3:5].set(rotate_xy(state[...,3:5],-turns))
    output=output.at[...,9:11].set(rotate_xy(state[...,9:11],-turns))
    sign=jnp.where(turns%2,-1,1)
    output=output.at[...,5].set(sign*state[...,5])
    return output.at[...,11].set(sign*state[...,11])


def rotate_obstacles(obstacles,turns):
    output=obstacles.at[...,0:2].set(rotate_xy(obstacles[...,0:2],turns))
    return output.at[...,3:5].set(rotate_xy(obstacles[...,3:5],turns))


def rotate_features(features,turns):
    """Transform one stored66x58 observation graph; masks and targets unchanged."""
    if features.shape[-1]!=58:raise ValueError('Expected the58-column observed Quad3D graph')
    result=features
    # Relative positions/velocities, goal/route target, and current velocity.
    for index in (3,5,9,12,21,52):
        result=result.at[...,index:index+2].set(rotate_xy(features[...,index:index+2],turns))
    # Pitch/roll, rates, and matching normalized observer-bias pairs.
    for index in (18,24,50,55):
        result=result.at[...,index:index+2].set(rotate_xy(features[...,index:index+2],-turns))
    sign=jnp.where(turns%2,-1,1)
    for index in (20,26,57):result=result.at[...,index].set(sign*features[...,index])
    return result.at[...,27:31].set(rotate_control(features[...,27:31],turns))


def augment_batch(batch,seed,step):
    """Same seeded transforms for both encoders; all branches share a rotation."""
    key=jax.random.fold_in(jax.random.PRNGKey(seed),step)
    turns=jax.random.randint(key,(len(batch['features']),),0,4)
    return dict(batch,features=jax.vmap(rotate_features)(batch['features'],turns))
