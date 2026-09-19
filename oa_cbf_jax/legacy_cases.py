"""Public comparison registry extracted unchanged from legacy media settings."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Case:
    group: str
    method: str
    robot: str
    controller: str
    checkpoint: str | None = None
    threshold: str | None = None

CASES = [
    Case("dynamic_unicycle", "fixed_low", "DynamicUnicycle2D", "MPC-CBF low fixed param"),
    Case("dynamic_unicycle", "fixed_high", "DynamicUnicycle2D", "MPC-CBF high fixed param"),
    Case("dynamic_unicycle", "od_cbf_qp", "DynamicUnicycle2D", "Optimal Decay CBF-QP"),
    Case("dynamic_unicycle", "od_cbf_mpc", "DynamicUnicycle2D", "Optimal Decay MPC-CBF"),
    Case("dynamic_unicycle", "barriernet", "DynamicUnicycle2D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/DynamicUnicycle2D_barriernet.pth"),
    Case("dynamic_unicycle", "ours_fc", "DynamicUnicycle2D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/DynamicUnicycle2D_1120_mlp_1230_epoch_400.pth", "0.130267"),
    Case("dynamic_unicycle", "ours_gat", "DynamicUnicycle2D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/DynamicUnicycle2D_1112_gat_2230_epoch_300.pth", "0.987007"),

    Case("quad2d", "fixed_low", "Quad2D", "MPC-CBF low fixed param"),
    Case("quad2d", "fixed_high", "Quad2D", "MPC-CBF high fixed param"),
    Case("quad2d", "od_cbf_qp", "Quad2D", "Optimal Decay CBF-QP"),
    Case("quad2d", "od_cbf_mpc", "Quad2D", "Optimal Decay MPC-CBF"),
    Case("quad2d", "barriernet", "Quad2D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/Quad2D_barriernet.pth"),
    Case("quad2d", "ours_fc", "Quad2D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/Quad2D_1120_mlp_1230_epoch_400.pth", "0.897910"),
    Case("quad2d", "ours_gat", "Quad2D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/Quad2D_1106_gat_2230_epoch_200.pth", "0.022786"),

    Case("quad3d", "fixed_low", "Quad3D", "MPC-CBF low fixed param"),
    Case("quad3d", "fixed_high", "Quad3D", "MPC-CBF high fixed param"),
    Case("quad3d", "od_cbf_mpc", "Quad3D", "Optimal Decay MPC-CBF"),
    Case("quad3d", "barriernet", "Quad3D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/Quad3D_barriernet.pth"),
    Case("quad3d", "ours_fc", "Quad3D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/Quad3D_1207_mlp_1230_epoch_800.pth", "13.608210"),
    Case("quad3d", "ours_gat", "Quad3D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/Quad3D_1103_gat_0230_epoch_600.pth", "0.877601"),

    Case("kinematic_bicycle_dpcbf", "fixed_low", "KinematicBicycle2D_DPCBF", "CBF-QP low fixed param"),
    Case("kinematic_bicycle_dpcbf", "fixed_high", "KinematicBicycle2D_DPCBF", "CBF-QP high fixed param"),
    Case("kinematic_bicycle_dpcbf", "od_cbf_qp", "KinematicBicycle2D_DPCBF", "Optimal Decay CBF-QP"),
    Case("kinematic_bicycle_dpcbf", "barriernet", "KinematicBicycle2D_DPCBF", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/KinematicBicycle2D_DPCBF_barriernet.pth"),
    Case("kinematic_bicycle_dpcbf", "ours_fc", "KinematicBicycle2D_DPCBF", "Online Adaptive CBF-QP MLP", "nn_model/checkpoint/KinematicBicycle2D_DPCBF_1207_mlp_1230_epoch_200.pth", "0.024335"),
    Case("kinematic_bicycle_dpcbf", "ours_gat", "KinematicBicycle2D_DPCBF", "Online Adaptive CBF-QP GAT", "nn_model/checkpoint/KinematicBicycle2D_DPCBF_1101_gat_1730_epoch_200.pth", "0.2930126190185547"),
]
