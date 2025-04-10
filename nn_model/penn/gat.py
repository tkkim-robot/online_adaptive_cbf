import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

from torch_scatter import scatter_sum, scatter_softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class GATModule(nn.Module):
    """
    - Create a dataset of dict items: {
        'robot': [x, y, vx, vy],
        'obstacles': [...],
        'goal': [gx, gy],
        'deadlock_time': float,
        'risk': float,
        'gamma0': float,
        'gamma1': float
    }
    - Wrap them via wrap_dataset_to_graphs(...).
    - Train with train_model(...).
    - Evaluate with evaluate_model(...).
    """

    def __init__(self, robot_radius=0.3, lr=0.001, num_epochs=50, batch_size=8, device='cpu'):
        super().__init__()
        self.robot_radius = robot_radius
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat = self.GATGraphNetwork().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.gat.parameters(), lr=self.lr)
        
    class GATGraphNetwork(nn.Module):
        """
        - z_ij = [v_i, v_j, e_ij]
        - Gains an attention-based node embedding
        - Final layer (psi4) is fed by:
            [robot_embedding(16) + gamma(2)] => 18 => 2D output => [deadlock time, risk level]
        """
        def __init__(self):
            super().__init__()

            # z_ij dimension: node features = 3 each (robot/obs/goal),
            # plus 5 for edge attributes => total 11
            self.zij_dim = 11

            # 1) psi1 => 16-dim edge embeddings
            self.psi1 = nn.Sequential(
                nn.Linear(self.zij_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16)
            )

            # 2) psi2 => attention weights
            self.psi2 = nn.Sequential(
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

            # 3) psi3 => transforms edge messages
            self.psi3 = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 16)
            )

            # 4) psi4 => final 2D output,
            # input = 16 (robot emb) + 2 (gamma0, gamma1) = 18
            self.psi4 = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # [deadlock time, risk level]
            )

        def forward(self, x, edge_index, edge_attr, batch, gammas=None):
            """
            Args:
              x: Node features [total_nodes, 3].
              edge_index: [2, num_edges], fully connected or partial graph.
              edge_attr: [num_edges, 5].
              batch: [total_nodes], indicates graph grouping in a batch.
              gammas: shape [num_graphs, 2] => (gamma0, gamma1) for each graph.

            Returns: [num_graphs, 2] => (deadlock_time, risk level).
            """
            device = next(self.parameters()).device  # 또는 x.device로 써도 됨
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)
            
            src, dst = edge_index
            v_i = x[src]
            v_j = x[dst]
            # Build z_ij => [num_edges, 11]
            zij = torch.cat([v_i, v_j, edge_attr], dim=1).to(device)

            # 1) psi1 => 16-dim edge embeddings
            q_ij = self.psi1(zij)

            # 2) attention weights
            raw_weights = self.psi2(q_ij).squeeze(-1)   # [num_edges]
            attn_weights = scatter_softmax(raw_weights, src)  # node-wise

            # 3) transform edge messages
            msg = self.psi3(q_ij)  # [E, 16]
            weighted = attn_weights.unsqueeze(-1) * msg

            # node embedding via scatter sum => [total_nodes, 16]
            node_emb = scatter_sum(weighted, src, dim=0)

            # find the "robot node" index for each sub-graph => first node in each group
            batch = batch.to(device)
            first_nodes = torch.cat((
                torch.tensor([0], device=device),
                (torch.where(torch.diff(batch))[0] + 1).to(device)
            ))            
            # first_nodes = torch.cat((
            #     torch.tensor([0], device=device),
            #     torch.where(torch.diff(batch))[0] + 1
            # ))
            robot_q = node_emb[first_nodes]  # [num_graphs, 16]

            gammas = gammas.to(device)
            combined = torch.cat([robot_q, gammas], dim=1)

            # 4) final => 2D => [deadlock, risk]
            out_2d = self.psi4(combined)
            return out_2d

        def extract_robot_embedding(self, x, edge_index, edge_attr, batch):
            """
            Returns the 16D robot embedding only,
            ignoring the final psi4 layer.
            """
            device = next(self.parameters()).device  # 또는 x.device로 써도 됨
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)
            
            src, dst = edge_index
            v_i = x[src]
            v_j = x[dst]
            zij = torch.cat([v_i, v_j, edge_attr], dim=1).to(device)  # [E, 11]

            q_ij = self.psi1(zij)  # [E,16]

            raw_weights = self.psi2(q_ij).squeeze(-1)
            attn_weights = scatter_softmax(raw_weights, src)
            messages = self.psi3(q_ij)
            weighted = attn_weights.unsqueeze(-1)*messages
            node_emb = scatter_sum(weighted, src, dim=0)

            # find first node => robot
            batch = batch.to(device)
            first_nodes = torch.cat((
                torch.tensor([0], device=device),
                (torch.where(torch.diff(batch))[0] + 1).to(device)
            ))
            
            # first_nodes = torch.cat((
            #     torch.tensor([0], device=node_emb.device),
            #     torch.where(torch.diff(batch))[0] + 1
            # ))
            
            robot_emb = node_emb[first_nodes]  # shape = [num_graphs,16]
            return robot_emb


    def create_graph(self, robot, obstacles, goal, deadlock=0.0, risk=0.0):
        """
        Builds a PyG Data object. We store a 2D target => [deadlock, risk].
        robot = [x, y, vx, vy]
        obstacles = list of [ox, oy, radius]
        goal = [gx, gy]
        """
        from torch_geometric.data import Data

        # 1) Construct node features
        num_obstacles = len(obstacles)
        num_nodes = num_obstacles + 2  # 1 robot + N obstacles + 1 goal

        # One-hot node classes
        robot_feature = [1, 0, 0]
        obs_feature   = [0, 1, 0]
        goal_feature  = [0, 0, 1]

        node_features = [robot_feature] + [obs_feature]*num_obstacles + [goal_feature]
        node_features = torch.tensor(node_features, dtype=torch.float)

        # 2) Positions
        robot_pos = torch.tensor(robot[:2], dtype=torch.float).unsqueeze(0)
        obs_pos   = torch.tensor([obs[:2] for obs in obstacles], dtype=torch.float) if num_obstacles>0 else torch.empty(0,2)
        goal_pos  = torch.tensor(goal, dtype=torch.float).unsqueeze(0)
        all_positions = torch.cat([robot_pos, obs_pos, goal_pos], dim=0)

        # 3) Radii
        radii_list = [self.robot_radius] + [obs[2] for obs in obstacles] + [0.0]
        radii = torch.tensor(radii_list, dtype=torch.float)

        # 4) Fully connected edges
        edge_index_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index_list.append([i, j])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()

        # 5) Relative positions => [N, N, 2]
        rel_pos = -all_positions.unsqueeze(1) + all_positions.unsqueeze(0)
        dist = torch.norm(rel_pos, dim=2)
        radii_mat = radii.unsqueeze(0) + radii.unsqueeze(1)
        dist = torch.clamp(dist - radii_mat, min=0.0).unsqueeze(-1)

        # 6) Velocities
        robot_vel = torch.tensor(robot[2:], dtype=torch.float).unsqueeze(0)
        obs_vel   = torch.zeros((num_obstacles, 2), dtype=torch.float)
        goal_vel  = torch.zeros((1,2), dtype=torch.float)
        all_vels  = torch.cat([robot_vel, obs_vel, goal_vel], dim=0)
        rel_vel   = -all_vels.unsqueeze(1) + all_vels.unsqueeze(0) # [N, N, 2]

        # 7) Combine => [dx, dy, distance, dvx, dvy]
        edge_features = torch.cat([rel_pos, dist, rel_vel], dim=2).view(-1, 5)
        src_lin = edge_index[0] * num_nodes + edge_index[1]
        edge_attr = edge_features[src_lin]

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor([[deadlock, risk]], dtype=torch.float)  # 2D label
        return data

    def wrap_dataset_to_graphs(self, dataset):
        """
        Expects each sample to have keys like:
            'robot': [x, y, vx, vy]
            'obstacles': ...
            'goal': [gx, gy]
            'deadlock_time': float
            'risk': float
            'gamma0': float
            'gamma1': float
        """
        graph_list = []
        for sample in dataset:
            deadlock_val = sample.get('deadlock_time', 0.0)
            risk_val     = sample.get('risk', 0.0)
            g = self.create_graph(
                robot=sample['robot'],
                obstacles=sample['obstacles'],
                goal=sample['goal'],
                deadlock=deadlock_val,
                risk=risk_val
            )
            gamma0 = sample.get('gamma0', 0.0)
            gamma1 = sample.get('gamma1', 0.0)
            g.gamma = torch.tensor([[gamma0, gamma1]], dtype=torch.float)

            graph_list.append(g)
        return graph_list

    def create_dataloader(self, graphs_dataset, batch_size=32, shuffle=True):
        return DataLoader(graphs_dataset, batch_size=batch_size, shuffle=shuffle)

    def train_model(self, train_data, test_data):
        """
        Trains the GAT for 2D output: [deadlock, risk].
        We also feed each graph's gamma0,gamma1 into the forward pass.
        """
        train_loader = self.create_dataloader(train_data, shuffle=True)
        test_loader  = self.create_dataloader(test_data, shuffle=False)

        for epoch in range(self.num_epochs):
            self.gat.train()
            total_loss = 0.0
            for batch in train_loader:
                # # batch: a collated PyG Data object
                # x, edge_index, edge_attr, y, batch_vec = (
                #     batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                # )
                # # gammas => shape [num_graphs,2]
                # gammas = getattr(batch, 'gamma', None)
                
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device)
                y = batch.y.to(self.device)
                batch_vec = batch.batch.to(self.device)
                gammas = getattr(batch, 'gamma', None)
                gammas = gammas.to(self.device)
                
                pred_2d = self.gat(x, edge_index, edge_attr, batch_vec, gammas)

                loss = self.criterion(pred_2d, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # validation
            self.gat.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    # x, edge_index, edge_attr, y, batch_vec = (
                    #     batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                    # )
                    # gammas = getattr(batch, 'gamma', None)
                    
                    x = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                    edge_attr = batch.edge_attr.to(self.device)
                    y = batch.y.to(self.device)
                    batch_vec = batch.batch.to(self.device)
                    gammas = getattr(batch, 'gamma', None)
                    gammas = gammas.to(self.device)
                    
                    pred_2d = self.gat(x, edge_index, edge_attr, batch_vec, gammas)
                    val_loss += self.criterion(pred_2d, y).item()

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    def evaluate_model(self, data_list):
        """
        Evaluates the GAT on 2D output: [deadlock, risk].
        """
        loader = self.create_dataloader(data_list, shuffle=False)
        self.gat.eval()

        preds_list, targets_list = [], []
        with torch.no_grad():
            for batch in loader:
                # x, edge_index, edge_attr, y, batch_vec = (
                #     batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                # )
                # gammas = getattr(batch, 'gamma', None)
                
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device)
                y = batch.y.to(self.device)
                batch_vec = batch.batch.to(self.device)
                gammas = getattr(batch, 'gamma', None)
                gammas = gammas.to(self.device)                
                
                pred_2d = self.gat(x, edge_index, edge_attr, batch_vec, gammas)
                preds_list.append(pred_2d.cpu().numpy())
                targets_list.append(y.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)   # shape [N, 2]
        targets = np.concatenate(targets_list, axis=0) # shape [N, 2]

        deadlock_mse = mean_squared_error(targets[:,0], preds[:,0])
        risk_mse     = mean_squared_error(targets[:,1], preds[:,1])
        deadlock_mae = mean_absolute_error(targets[:,0], preds[:,0])
        risk_mae     = mean_absolute_error(targets[:,1], preds[:,1])
        deadlock_r2  = r2_score(targets[:,0], preds[:,0])
        risk_r2      = r2_score(targets[:,1], preds[:,1])

        return {
            "MSE_deadlock": deadlock_mse,
            "MSE_risk": risk_mse,
            "MAE_deadlock": deadlock_mae,
            "MAE_risk": risk_mae,
            "R2_deadlock": deadlock_r2,
            "R2_risk": risk_r2
        }

    def visualize_graph(self, graph):
        """
        Converts a single PyG Data object to a NetworkX graph and plots it.
        Edge attributes are labeled with the third element (distance) for demonstration.
        """
        nx_graph = to_networkx(graph, edge_attrs=["edge_attr"])
        plt.figure(figsize=(5,5))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")

        # Show distance on edges => index 2 in edge_attr is the distance
        edge_labels = nx.get_edge_attributes(nx_graph, "edge_attr")
        labeled = {k: f"{v[2]:.2f}" for k,v in edge_labels.items()}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labeled, font_size=8)
        plt.title("Graph Visualization")
        plt.show()


def demo():
    """
    Demonstrates usage of the revised GATModule:
      - Each sample has deadlock_time, risk, gamma0, gamma1
      - GAT predicts 2D: [deadlock, risk]
      - gamma0, gamma1 are concatenated with the robot embedding
    """
    module = GATModule(lr=0.00005, num_epochs=2000, batch_size=32)

    # Build a small random dataset
    dataset = []
    for _ in range(30):
        robot_state = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0)
        ]
        n_obs = np.random.randint(2, 5)
        obstacles = []
        for __ in range(n_obs):
            ox = np.random.uniform(0.5, 1.5)
            oy = np.random.uniform(0.5, 1.5)
            r  = np.random.uniform(0.1, 0.3)
            obstacles.append([ox, oy, r])
        goal = [1.5, 1.5]

        deadlock_t = np.random.uniform(0.0, 5.0)
        risk_val   = np.random.uniform(0.0, 2.0)
        gamma0     = np.random.uniform(0.01, 1.0)
        gamma1     = np.random.uniform(0.01, 1.0)

        dataset.append({
            'robot': robot_state,
            'obstacles': obstacles,
            'goal': goal,
            'deadlock_time': deadlock_t,
            'risk': risk_val,
            'gamma0': gamma0,
            'gamma1': gamma1
        })

    # Convert to PyG Data
    graphs = module.wrap_dataset_to_graphs(dataset)

    # Train-test split
    train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)

    # Train
    module.train_model(train_data, test_data)

    # Evaluate
    metrics = module.evaluate_model(test_data)
    print("Evaluation metrics:", metrics)

    # Visualize one sample
    module.visualize_graph(train_data[0])


if __name__ == "__main__":
    demo()
