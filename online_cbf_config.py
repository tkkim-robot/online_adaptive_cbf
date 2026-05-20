import numpy as np

pillar_1_x = 67.0
pillar_2_x = 73.0

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
                "gamma0": 0.35,
                "gamma1": 0.35
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
            "Online Adaptive CBF-QP": {
                "type": "cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5
            },
            "Online Adaptive MPC-CBF MLP": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "Online Adaptive MPC-CBF GAT": {
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

    "KinematicBicycle2D_DPCBF": {
        "controller_params": {
            "CBF-QP low fixed param": {
                "type": "cbf_qp",
                "gamma0": 0.1,
                "gamma1": 0.0
            },
            "CBF-QP high fixed param": {
                "type": "cbf_qp",
                "gamma0": 70.0,
                "gamma1": 0.0
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.1,
                "gamma1": 0.0
            },
            "Online Adaptive CBF-QP MLP": {
                "type": "cbf_qp",
                "gamma0": 0.1,
                "gamma1": 0.0
            },
            "Online Adaptive CBF-QP GAT": {
                "type": "cbf_qp",
                "gamma0": 0.1,
                "gamma1": 0.0
            },
        },
        "robot_spec": {
            "model": "KinematicBicycle2D_DPCBF",
            "a_max": 5.0,
            "radius": 0.3,
        },
        "default_obs": np.array([
            [4.0, 0.1, 0.3],
            [3.5, 0.3, 0.4],
            [2.5, 3.0, 0.5],
            [6.5, 2.4, 1.05],
            [7.5, 2.7, 0.45],
            [8.2, 0.7, 0.5],
        ])
    },

    "Quad2D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01,
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.99,
                "gamma1": 0.99,
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5,
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01,
            },
            "Online Adaptive CBF-QP": {
                "type": "cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5,
            },
            "Online Adaptive MPC-CBF MLP": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01,
            },
            "Online Adaptive MPC-CBF GAT": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01,
            },
        },
        "robot_spec": {
            "model": "Quad2D",
            "f_min": 2.5,
            "f_max": 5.5,
            "inertia": 0.05,
            "sensor": "rgbd",
            "radius": 0.3
        },
        "default_obs": np.array([
            [4.0, 0.1, 0.3],
            [3.5, 0.3, 0.4],
            [2.5, 3.0, 0.5],
            [6.5, 2.4, 1.05],
            [7.5, 2.7, 0.45],
            [8.2, 0.7, 0.5],
        ])
    },

    "Quad3D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.99,
                "gamma1": 0.99
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "Online Adaptive MPC-CBF MLP": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
            "Online Adaptive MPC-CBF GAT": {
                "type": "mpc_cbf",
                "gamma0": 0.01,
                "gamma1": 0.01
            },
        },
        "robot_spec": {
            "model": "Quad3D",
            "u_min": -2.0,
            "u_max": 5.0,
            "sensor": "rgbd",
            "radius": 0.3
        },
        "default_obs": np.array([
            [0.1, 2.0, 0.1],
            [0.4, 2.7, 0.1],
            # [0.5, 3.0, 0.1],
            [0.5, 0.5, 0.2],
            [1.7, 3.2, 0.5],
            [2.0, 0.4, 0.3],
            [3.1, 2.2, 0.5],
            [3.5, 0.6, 0.4],
            [4.6, 2.7, 0.6],
            [4.9, 0.9, 0.4],
            [6.8, 2.5, 1.0],
            [6.2, 0.3, 0.25],
            [7.6, 0.5, 0.25],
            # [7.5, 2.5, 0.45],
            [8.7, 0.8, 0.5],
        ])
    },

    "VTOL2D": {
        "controller_params": {
            "MPC-CBF low fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.05,
                "gamma1": 0.05
            },
            "MPC-CBF high fixed param": {
                "type": "mpc_cbf",
                "gamma0": 0.35,
                "gamma1": 0.35
            },
            "Optimal Decay CBF-QP": {
                "type": "optimal_decay_cbf_qp",
                "gamma0": 0.5,
                "gamma1": 0.5
            },
            "Optimal Decay MPC-CBF": {
                "type": "optimal_decay_mpc_cbf",
                "gamma0": 0.05,
                "gamma1": 0.05
            },
            "Online Adaptive CBF MLP": {
                "type": "mpc_cbf",
                "gamma0": 0.05,
                "gamma1": 0.05
            },
        },
        "robot_spec": {
            'model': "VTOL2D",
            'radius': 0.6,
            'v_max': 20.0,
            'reached_threshold': 1.0 # meter
        },
        "default_obs": np.array([
            [pillar_1_x, 6.0, 0.5],
            [pillar_1_x, 7.0, 0.5],
            [pillar_1_x, 8.0, 0.5],
            [pillar_1_x, 9.0, 0.5],
            [pillar_2_x, 1.0, 0.5],
            [pillar_2_x, 2.0, 0.5],
            [pillar_2_x, 3.0, 0.5],
            [pillar_2_x, 4.0, 0.5],
            [pillar_2_x, 5.0, 0.5],
            [pillar_2_x, 6.0, 0.5],
            [pillar_2_x, 7.0, 0.5],
            [pillar_2_x, 8.0, 0.5],
            [pillar_2_x, 9.0, 0.5],
            [pillar_2_x, 10.0, 0.5],
            [pillar_2_x, 11.0, 0.5],
            [pillar_2_x, 12.0, 0.5],
            [pillar_2_x, 13.0, 0.5],
            [pillar_2_x, 14.0, 0.5],
            [pillar_2_x, 15.0, 0.5],
            [60.0, 12.0, 1.5]
            ]),
        "env_width": 75.0,
        "env_height": 15.0
    }
}

# Config for the online adapter (model file paths, step sizes, etc.)
ADAPTIVE_MODELS = {
    "DynamicUnicycle2D": {
        "online_cbf_qp": {
            "model_path":  "nn_model/checkpoint/DynamicUnicycle2D_0530_mlp_qp.pth",
            "scaler_path": "nn_model/checkpoint/DynamicUnicycle2D_0530_mlp_qp.save",
            "step_size":   0.05,
            "lower_bound": 0.5,
            "upper_bound": 3.0,
            "raw_epistemic_threshold": 0.1            
        },
        "online_mpc_cbf_mlp": {
            "model_path":  "nn_model/checkpoint/DynamicUnicycle2D_0801_mlp_0230.pth",
            "scaler_path": "nn_model/checkpoint/DynamicUnicycle2D_0801_mlp_0230.save",
            # "model_path":  "nn_model/checkpoint/DynamicUnicycle2D_0418_mlp.pth",
            # "scaler_path": "nn_model/checkpoint/DynamicUnicycle2D_0418_mlp.save",
            "step_size":   0.01,
            "lower_bound": 0.01,
            "upper_bound": 0.35,
            "raw_epistemic_threshold": 0.1            
        },      
        "online_mpc_cbf_gat": {
            # "model_path":  "nn_model/checkpoint/DynamicUnicycle2D_0808_gat_1730.pth",
            # "scaler_path": "nn_model/checkpoint/DynamicUnicycle2D_0808_gat_1730.save",
            "model_path":  "nn_model/checkpoint/DynamicUnicycle2D_0731_gat_2130.pth",
            "scaler_path": "nn_model/checkpoint/DynamicUnicycle2D_0731_gat_2130.save",
            "step_size":   0.01,
            "lower_bound": 0.01,
            "upper_bound": 0.35,
            "raw_epistemic_threshold": 0.1          
        }      
    },
    "KinematicBicycle2D_DPCBF": {
        "online_cbf_qp": {
            "model_path":  "nn_model/checkpoint/penn_model_qp.pth",
            "scaler_path": "nn_model/checkpoint/scaler_qp.save",
            "step_size":   0.01,
            "lower_bound": 0.1,
            "upper_bound": 0.2
        },
        "online_cbf_qp_mlp": {
            "model_path":  "nn_model/checkpoint/penn_model_qp.pth",
            "scaler_path": "nn_model/checkpoint/scaler_qp.save",
            "step_size":   0.05,
            "lower_bound": 0.1,
            "upper_bound": 15.0
        },
        "online_cbf_qp_gat": {
            "model_path":  "nn_model/checkpoint/KinematicBicycle2D_DPCBF_1101_gat_1730_epoch_200.pth",
            "scaler_path": None,
            "step_size":   0.05,
            "lower_bound": 0.1,
            "upper_bound": 15.0,
            "raw_epistemic_threshold": 0.2930126190185547
        },
        "online_mpc_cbf_mlp": {
            "model_path":  "nn_model/checkpoint/KinematicBicycle2D_C3BF_0621_mlp_2430.pth",
            "scaler_path": "nn_model/checkpoint/KinematicBicycle2D_C3BF_0621_mlp_2430.save",
            "step_size":   0.005,
            "lower_bound": 0.01,
            "upper_bound": 0.99
        },
        "online_mpc_cbf_gat": {
            "model_path":  "nn_model/checkpoint/KinematicBicycle2DDP_0817_gat_0130.pth",
            "scaler_path": "nn_model/checkpoint/KinematicBicycle2DDP_0814_gat_0130.save",
            "step_size":   0.005,
            "lower_bound": 0.01,
            "upper_bound": 0.99
        }  
    },
    "Quad2D": {
        "online_cbf_qp": {
            "model_path":  "nn_model/checkpoint/penn_model_qp.pth",
            "scaler_path": "nn_model/checkpoint/scaler_qp.save",
            "step_size":   0.01,
            "lower_bound": 0.01,
            "upper_bound": 0.2
        },
        "online_mpc_cbf_mlp": {
            # "model_path":  "nn_model/checkpoint/Quad2D_0804_mlp_2330_2.pth",
            "model_path":  "nn_model/checkpoint/Quad2D_0804_mlp_2330_2.pth",
            "scaler_path": "nn_model/checkpoint/Quad2D_0804_mlp_2330.save",
            # "model_path":  "nn_model/checkpoint/Quad2D_0117_mlp.pth",
            # "scaler_path": "nn_model/checkpoint/Quad2D_0117_mlp.save",
            "step_size":   0.04,
            "lower_bound": 0.01,
            "upper_bound": 0.99,
            "raw_epistemic_threshold": 0.2            
        },  
        "online_mpc_cbf_gat": {
            "model_path":  "nn_model/checkpoint/Quad2D_0804_gat_1930.pth",
            "scaler_path": "nn_model/checkpoint/Quad2D_0804_gat_1930.save",
            # "model_path":  "nn_model/checkpoint/Quad2D_0514_gat.pth",
            # "scaler_path": "nn_model/checkpoint/Quad2D_0514_gat.save",
            "step_size":   0.04,
            "lower_bound": 0.01,
            "upper_bound": 0.99,
            "raw_epistemic_threshold": 0.2            
        },
    },
    "Quad3D": {
        "online_cbf_qp": {
            "model_path":  "nn_model/checkpoint/penn_model_qp.pth",
            "scaler_path": "nn_model/checkpoint/scaler_qp.save",
            "step_size":   0.01,
            "lower_bound": 0.01,
            "upper_bound": 0.2
        },
        "online_mpc_cbf_mlp": {
            "model_path":  "nn_model/checkpoint/Quad3D_0808_mlp_1230.pth",
            "scaler_path": "nn_model/checkpoint/Quad3D_0808_mlp_1230.save",
            # "model_path":  "nn_model/checkpoint/Quad3D_0728_mlp_1230.pth",
            # "scaler_path": "nn_model/checkpoint/Quad3D_0728_mlp_1230.save",
            # "model_path":  "nn_model/checkpoint/Quad3D_0708_mlp_1130.pth",
            # "scaler_path": "nn_model/checkpoint/Quad3D_0708_mlp_1130.save",
            "step_size":   0.003,
            "lower_bound": 0.01,
            "upper_bound": 0.99,
            "raw_epistemic_threshold": 0.2            
        },  
        "online_mpc_cbf_gat": {
            # "model_path":  "nn_model/checkpoint/Quad3D_0807_gat_0230.pth",
            # # "model_path":  "nn_model/checkpoint/best_gat_penn.pth",
            # "scaler_path": "nn_model/checkpoint/Quad3D_0807_gat_0230.save",
            "model_path":  "nn_model/checkpoint/Quad3D_1103_gat_0230_epoch_600.pth",
            # "model_path":  "nn_model/checkpoint/best_gat_penn.pth",
            "scaler_path": "nn_model/checkpoint/Quad3D_1103_gat_0230_epoch_600.save",
            "step_size":   0.003,
            "lower_bound": 0.01,
            "upper_bound": 0.99,
            "raw_epistemic_threshold": 0.6776006698608398
        },
    },
    
    "VTOL2D": {
        "model_path":  "nn_model/checkpoint/penn_model_vtol_0224.pth",
        "scaler_path": "nn_model/checkpoint/scaler_vtol_0224.save",
        "step_size":   0.01,
        "lower_bound": 0.05,
        "upper_bound": 0.35,
        "raw_epistemic_threshold": 0.05
    }
}
