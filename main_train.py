import argparse

import torch

from samp.model import MotionPlanningNetwork
from samp.trainer import Trainer
from samp.dataset import CustomDataset
from samp.loss import MathcLossAndCollsionLoss

from encode_panda_sdf_points_sampler import HashMLPModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_dataset = CustomDataset(args.datset_path, args.env_dataset_path, args.anchor_size,)

    loss_fn = MathcLossAndCollsionLoss()

    # Motion generation
    model = MotionPlanningNetwork(args.q_feature_size, args.anchor_size, args.output_size)
    model = model.to(device)

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of the parameters of the neural network: {total_parameters}')
    trainer = Trainer(custom_dataset, model, loss_fn, args.initial_lr, args.num_epochs, args.batch_size, args.weight_decay)
    trainer.train()
    # print(trainer.test_overfit_single_batch())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datset_path', type=str, default='diverse_processed_dataset.hdf5', help='Path to the dataset file')
    parser.add_argument('--env_dataset_path', type=str, default='diverse_processed_env.hdf5', help='Path to the environment dataset file')
    parser.add_argument('--q_feature_size', type=int, default=14, help='Size of the q feature vector, q_current + q_goal')
    parser.add_argument('--anchor_size', type=int, default=100, help='Size of the anchor points grid')
    parser.add_argument('--output_size', type=int, default=7, help='Output size of the model')

    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')

    args = parser.parse_args()
    main(args)