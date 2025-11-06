import torch
import torch.nn.functional as F

from samp.utils import grid_normalize_points

# def point_match_loss(pred_robot_sdf, next_robot_sdf, boundary=0.1, reduction="mean"):
#     mask = ((pred_robot_sdf.abs() < boundary) | (next_robot_sdf.abs() < boundary))
#     if mask.sum() == 0:
#         return torch.tensor(0.0, device=input_robot_sdf.device)
#     input_sel = pred_robot_sdf[mask]
#     target_sel = next_robot_sdf[mask]
#     return F.mse_loss(input_sel, target_sel, reduction=reduction) + F.l1_loss(input_sel, target_sel, reduction=reduction)

def point_match_loss(pred_robot_sdf, next_robot_sdf, reduction="mean"):
    input_sel = pred_robot_sdf
    target_sel = next_robot_sdf

    return F.mse_loss(input_sel, target_sel, reduction=reduction) + F.l1_loss(input_sel, target_sel, reduction=reduction)

def collision_loss(
    pred_robot_points: torch.Tensor,
    pred_robot_sdf: torch.Tensor,
    env_sdf: torch.Tensor,
    target_distance: float = 0.015,
    # d_safe: float = 0.01
):
    pred_robot_points = pred_robot_points.unsqueeze(1).unsqueeze(1) 
    pred_robot_sdf = pred_robot_sdf.unsqueeze(1).unsqueeze(1).unsqueeze(1) 

    pred_robot_points = grid_normalize_points(pred_robot_points)
    swap_idx = [2, 1, 0]  # z, y, x
    pred_robot_points_swapped = pred_robot_points[..., swap_idx]

    assigned_collision = F.grid_sample(env_sdf, pred_robot_points_swapped, mode='bilinear', padding_mode='border', align_corners=True)
    weights = torch.exp(-1.0 * (pred_robot_sdf - target_distance))
    
    point_loss = F.hinge_embedding_loss(
        assigned_collision,
        -torch.ones_like(assigned_collision),
        margin=0.09,
        reduction="mean"
    )

    weighted_loss = (weights * point_loss).mean()
    return weighted_loss

def q_loss(
        pred_joint_state: torch.Tensor,
        next_joint_state: torch.Tensor,
        reduction: str = "mean"
):
    return F.mse_loss(pred_joint_state, next_joint_state, reduction=reduction) # + F.l1_loss(pred_joint_state, next_joint_state, reduction=reduction)



def final_q_loss(
        pred_joint_state: torch.Tensor,
        goal_joint_state: torch.Tensor,
        reduction: str = "mean"
):
    return F.l1_loss(pred_joint_state, goal_joint_state, reduction=reduction) + \
              F.mse_loss(pred_joint_state, goal_joint_state, reduction=reduction)


class MathcLossAndCollsionLoss:
    def __init__(
        self, d_safe: float = 0.1
    ):
        self.d_safe = d_safe


    def __call__(self,
        pred_joint_state: torch.Tensor,
        goal_joint_state: torch.Tensor, # final expert robot configuration
        next_joint_state: torch.Tensor, # next expert robot configuration q+1
        pred_robot_points: torch.Tensor, # current q robot sdf value
        next_robot_points: torch.Tensor, # next q robot sdf value
        pred_robot_sdf: torch.Tensor, # current q robot sdf value
        env_sdf: torch.Tensor, # environment sdf value
    ):
        match_loss = point_match_loss(pred_robot_points, next_robot_points)
        coll_loss = collision_loss(pred_robot_points, pred_robot_sdf, env_sdf)
        config_loss = q_loss(pred_joint_state, next_joint_state)
        final_loss = final_q_loss(pred_joint_state, goal_joint_state)

        return match_loss, coll_loss, config_loss, final_loss
        

if __name__ == "__main__":
    # Example usage
    input_robot_sdf = torch.randn(10, 1, 16, 16, 16) 
    next_robot_sdf = torch.randn(10, 1, 16, 16, 16)
    input_env_sdf = torch.randn(10, 1, 16, 16, 16)

    pred_joint_state = torch.randn(10, 7)  # Predicted joint state
    goal_joint_state = torch.randn(10, 7)  # Goal joint state
    next_joint_state = torch.randn(10, 7)  # Next joint state

    loss_fn = MathcLossAndCollsionLoss(d_safe=0.01)
    match_loss, coll_loss, config_loss, final_loss = loss_fn(pred_joint_state, goal_joint_state, next_joint_state,
                        input_robot_sdf, input_env_sdf, next_robot_sdf)

    print(f"Match Loss: {match_loss.item()}, Collision Loss: {coll_loss.item()}, Config Loss: {config_loss.item()}, Final Loss: {final_loss.item()}")