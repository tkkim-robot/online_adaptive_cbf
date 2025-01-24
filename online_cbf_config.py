import numpy as np

ALL_DEFAULTS = {
    "DynamicUnicycle2D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.2,
                "gamma1": 0.2
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "Online Adaptive CBF": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
        },
        "robot_spec": {
            "model": "DynamicUnicycle2D",
            "w_max": 0.5,
            "a_max": 0.5,
            "fov_angle": 70.0,
            "cam_range": 3.0,
            "radius": 0.3
        },
        "default_obs": np.array([
            [4.0, 0.3, 0.3],
            [3.5, 0.5, 0.4],
            [3.5, 2.4, 0.5],
            [6.5, 2.6, 1.05],
            [8.5, 0.4, 0.2],
            [8,   0.6, 0.35],
            [7.5, 2.3, 0.45],
        ])
    },

    "KinematicBicycle2D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.05,
                "gamma1": 0.05
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 3.0,
                "gamma1": 3.0
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.1,
                "gamma1": 0.1
            },
            "Online Adaptive CBF": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
        },
        "robot_spec": {
            "model": "KinematicBicycle2D",
            "a_max": 0.5,
            "fov_angle": 170.0,
            "cam_range": 0.01,
            "radius": 0.5
        },
        "default_obs": np.array([
            [4.0, 0.3, 0.25],
            [3.5, 0.5, 0.35],
            [3.5, 2.4, 0.45],
            [6.5, 2.6, 1.0],
            [8.2, 1.25, 0.25],
            [7.5, 2.3, 0.4],
        ])
    },

    "Quad2D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 1.1,
                "gamma1": 1.1
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "Online Adaptive CBF": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
        },
        "robot_spec": {
            "model": "Quad2D",
            "f_min": 3.0,
            "f_max": 10.0,
            "sensor": "rgbd",
            "radius": 0.25
        },
        "default_obs": np.array([
            [4.0, 0.1, 0.3],
            [3.5, 0.3, 0.4],
            [3.5, 3.0, 0.5],
            [6.5, 2.4, 1.05],
            [7.5, 2.7, 0.45],
            [8.2, 1.2, 0.5],
        ])
    }
}

# Config for the online adapter (model file paths, step sizes, etc.)
ADAPTIVE_MODELS = {
    "DynamicUnicycle2D": {
        "model_path":  "nn_model/checkpoint/penn_model_0921.pth",
        "scaler_path": "nn_model/checkpoint/scaler_0921.save",
        "step_size":   0.01,
        "lower_bound": 0.01,
        "upper_bound": 0.2
    },
    "KinematicBicycle2D": {
        "model_path":  "nn_model/checkpoint/penn_model_1204_kinbi.pth",
        "scaler_path": "nn_model/checkpoint/scaler_1204_kinbi.save",
        "step_size":   0.05,
        "lower_bound": 0.01,
        "upper_bound": 3.0
    },
    "Quad2D": {
        "model_path":  "nn_model/checkpoint/penn_model_0114_quad.pth",
        "scaler_path": "nn_model/checkpoint/scaler_0114_quad.save",
        "step_size":   0.05,
        "lower_bound": 0.01,
        "upper_bound": 1.1
    }
}
