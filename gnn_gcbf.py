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



class GCBFModule(nn.Module):
    """
    A class that encapsulates:
      1) Graph building (create_graph, wrap_dataset_to_graphs)
      2) The GNN model (GCBFGraphNetwork)
      3) Data loading (create_dataloader)
      4) Training (train_model)
      5) Evaluation (evaluate_model)
      6) Graph visualization (visualize_graph)

    Usage Example (in another file):
    -----------------------------------------------------------------------
    from gnn_gcbf import GCBFModule

    module = GCBFModule()
    # Suppose dataset is a list of dicts with keys: 'robot', 'obstacles', 'goal', 'risk'
    graphs_dataset = module.wrap_dataset_to_graphs(dataset)
    train_data, test_data = train_test_split(graphs_dataset, test_size=0.2)

    module.train_model(train_data, test_data)
    metrics = module.evaluate_model(test_data)
    print("Metrics:", metrics)
    -----------------------------------------------------------------------
    """

    def __init__(self, robot_radius=0.05, lr=0.001, num_epochs=50, batch_size=8):
        """
        The constructor initializes the internal GCBFGraphNetwork and sets default hyperparameters.

        Args:
            robot_radius (float): The radius of the robot circle for distance adjustments.
            lr (float): Learning rate for Adam optimizer.
            num_epochs (int): Default number of epochs for training.
            batch_size (int): Default batch size for data loading.
        """
        super().__init__()
        self.robot_radius = robot_radius
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize the GCBF-inspired GNN
        self.gnn = self.GCBFGraphNetwork()

        # Define optimizer and loss function
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)


    class GCBFGraphNetwork(nn.Module):
        """
        GCBF-inspired Graph Neural Network.

        - For each edge: z_ij = [v_i, v_j, e_ij].
        - Compute attention weights with node-wise softmax.
        - Aggregate messages at each node.
        - Final output is a scalar from the 'robot' node embedding of each graph.
        """
        def __init__(self):
            super().__init__()
            # Node features are 3D (one-hot for [robot, obs, goal]),
            # so for an edge: v_i + v_j => 6D, plus edge_attr => 5D => total 11.
            self.zij_dim = 11

            # MLP ψθ1: Encodes z_ij
            self.psi1 = nn.Sequential(
                nn.Linear(self.zij_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

            # MLP ψθ2: Computes attention weights
            self.psi2 = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

            # MLP ψθ3: Transforms edge-specific messages
            self.psi3 = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

            # MLP ψθ4: Outputs final scalar (y_hat)
            self.psi4 = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, x, edge_index, edge_attr, batch):
            """
            Args:
                x: [total_nodes, 3] => node features
                edge_index: [2, num_edges]
                edge_attr: [num_edges, 5]
                batch: [total_nodes] => indicates which graph each node belongs to

            Returns:
                [num_graphs, 1] => predicted scalar per graph
            """
            # Step 1: Create z_ij = [v_i, v_j, e_ij] for each edge
            src, dst = edge_index
            v_i = x[src]  # Features of source nodes [num_edges, 3]
            v_j = x[dst]  # Features of destination nodes [num_edges, 3]

            # Concatenate node + edge features => z_ij
            z_ij = torch.cat([v_i, v_j, edge_attr], dim=1)  # [num_edges, 11]


            # Step 2: Encode z_ij features via ψθ1
            q_ij = self.psi1(z_ij)  # [num_edges, 128]


            # Step 3: node-wise softmax !!
            # Compute attention weights => node-wise softmax by 'src'
            raw_weights = self.psi2(q_ij).squeeze(-1)  # [num_edges]
            attention_weights = scatter_softmax(raw_weights, src)  # [num_edges]

            # Transform edge messages
            messages = self.psi3(q_ij)  # [num_edges, 128]


            # Step 4: Aggregate messages to compute q_i (node embeddings) for each node
            # Weighted sum => node embeddings
            weighted_messages = attention_weights.unsqueeze(-1) * messages
            q_i = scatter_sum(weighted_messages, src, dim=0)  # [total_nodes, 128]

            # Extract the 'robot' node for each graph => index 0 in each sub-batch
            # Locate the first node of each graph
            robot_nodes = torch.cat((
                torch.tensor([0], device=batch.device),
                torch.where(torch.diff(batch))[0] + 1
            ))
            robot_q = q_i[robot_nodes]  # [num_graphs, 128]

            # Step 5: Pass the robot's feature vector through ψθ4
            output = self.psi4(robot_q)  # [num_graphs, 1]
            return output


    def create_graph(self, robot, obstacles, goal, risk=0.0):
        """
        Builds a PyTorch Geometric graph for the given scenario.

        Args:
            robot: [rx, ry, vx, vy]
            obstacles: list of [ox, oy, radius] for each obstacle
            goal: [gx, gy]
            risk: optional float for the ground-truth target (stored in data.y)
        """
        from torch_geometric.data import Data

        # Node features => robot (index 0), obstacles, goal (last index)
        num_obstacles = len(obstacles)
        num_nodes = num_obstacles + 2  # robot + obstacles + goal

        # One-hot node classes => [robot, obs, goal]
        robot_feature    = [1, 0, 0]
        obstacle_feature = [0, 1, 0]
        goal_feature     = [0, 0, 1]

        # Construct node feature matrix
        node_features = [robot_feature]
        node_features += [obstacle_feature] * num_obstacles
        node_features.append(goal_feature)
        node_features = torch.tensor(node_features, dtype=torch.float)  # [N, 3]

        # Positions
        robot_pos = torch.tensor(robot[:2], dtype=torch.float).unsqueeze(0)
        obstacle_pos = torch.tensor([obs[:2] for obs in obstacles], dtype=torch.float)
        goal_pos = torch.tensor(goal, dtype=torch.float).unsqueeze(0)
        all_positions = torch.cat([robot_pos, obstacle_pos, goal_pos], dim=0)  # [N, 2]

        # Radii
        radii_list = [self.robot_radius] + [obs[2] for obs in obstacles] + [0.0]
        radii = torch.tensor(radii_list, dtype=torch.float)  # [N]

        # Relative positions => shape [N, N, 2]
        rel_pos = -all_positions.unsqueeze(1) + all_positions.unsqueeze(0)
        dist    = torch.norm(rel_pos, dim=2)  # [N, N]
        radii_mat = radii.unsqueeze(0) + radii.unsqueeze(1)
        dist = torch.clamp(dist - radii_mat, min=0.0).unsqueeze(-1)  # [N, N, 1]

        # Relative velocities
        robot_vel = torch.tensor(robot[2:], dtype=torch.float).unsqueeze(0)
        obstacle_vel = torch.zeros((num_obstacles, 2), dtype=torch.float)
        goal_vel = torch.zeros((1, 2), dtype=torch.float)
        all_vels = torch.cat([robot_vel, obstacle_vel, goal_vel], dim=0)  # [N, 2]
        rel_vel  = -all_vels.unsqueeze(1) + all_vels.unsqueeze(0)         # [N, N, 2]

        # Fully connected edges => (i != j)
        edge_index_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index_list.append([i, j])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()  # [2, E]
        
        # Combine => [dx, dy, dist, dvx, dvy]
        edge_features = torch.cat([rel_pos, dist, rel_vel], dim=2)  # [N, N, 5]
        edge_features = edge_features.view(-1, 5)                   # [N*N, 5]

        # Map to actual edges
        src_lin = edge_index[0] * num_nodes + edge_index[1]
        edge_attr = edge_features[src_lin]  # [E, 5]

        # Build PyTorch Geometric Data object
        graph = Data(
            x=node_features,        # [N, 3]
            edge_index=edge_index,  # [2, E]
            edge_attr=edge_attr     # [E, 4]
        )
        graph.y = torch.tensor([risk], dtype=torch.float)
        return graph

    def wrap_dataset_to_graphs(self, dataset):
        """
        dataset => list of dict, each with keys:
            'robot': [x, y, vx, vy]
            'obstacles': list of [ox, oy, radius]
            'goal': [gx, gy]
            'risk': float
        Returns => list of PyG Data objects
        """
        graph_list = []
        for sample in dataset:
            graph = self.create_graph(
                    robot=sample['robot'],
                    obstacles=sample['obstacles'],
                    goal=sample['goal'],
                    risk=sample['risk']
                    )
            graph.y = torch.tensor([sample['risk']], dtype=torch.float)  # Add risk as target
            graph_list.append(graph)
        return graph_list


    def create_dataloader(self, graphs_dataset, batch_size=32, shuffle=True):
        """
        Returns a PyTorch Geometric DataLoader for the provided dataset.
        """
        return DataLoader(graphs_dataset, batch_size=batch_size, shuffle=shuffle)

    def train_model(self, train_data, test_data):
        """
        Basic MSE-based training loop.

        train_data, test_data => lists of PyG Data objects
        """
        train_loader = self.create_dataloader(train_data, shuffle=True)
        test_loader = self.create_dataloader(test_data, shuffle=False)

        for epoch in range(self.num_epochs):
            self.gnn.train()
            total_loss = 0.0
            for batch in train_loader:
                x, edge_index, edge_attr, y, batch_vec = (
                    batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                )
                pred = self.gnn(x, edge_index, edge_attr, batch_vec)
                loss = self.criterion(pred.squeeze(1), y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Validation
            self.gnn.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    x, edge_index, edge_attr, y, batch_vec = (
                        batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                    )
                    pred = self.gnn(x, edge_index, edge_attr, batch_vec)
                    val_loss += self.criterion(pred.squeeze(1), y).item()

            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    def evaluate_model(self, data_list):
        """
        Computes MSE, MAE, R^2 on the provided dataset.
        """
        loader = self.create_dataloader(data_list, shuffle=False)
        self.gnn.eval()

        preds_list, targets_list = [], []
        with torch.no_grad():
            for batch in loader:
                x, edge_index, edge_attr, y, batch_vec = (
                    batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                )
                pred = self.gnn(x, edge_index, edge_attr, batch_vec)
                preds_list.append(pred.detach().cpu().tolist())
                targets_list.append(y.detach().cpu().tolist())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        mse_val = mean_squared_error(targets, preds)
        mae_val = mean_absolute_error(targets, preds)
        r2_val  = r2_score(targets, preds)
        return {"MSE": mse_val, "MAE": mae_val, "R^2": r2_val}


    def visualize_graph(self, graph):
        """
        Uses NetworkX and matplotlib to visualize a PyTorch Geometric graph.
        """
        nx_graph = to_networkx(graph, edge_attrs=["edge_attr"])
        plt.figure(figsize=(5, 5))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")

        # Edge labels => e_ij: [dx, dy, dist, dvx, dvy]
        # Suppose index 2 is 'distance'
        edge_labels = nx.get_edge_attributes(nx_graph, "edge_attr")
        labeled = {k: f"{v[2]:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labeled, font_size=8)
        plt.title("Graph Visualization")
        plt.show()



def demo():
    """
    Demonstrates usage of the GCBFModule on a small dummy dataset.
    """
    module = GCBFModule(lr=0.001, num_epochs=50, batch_size=8)

    # Create a small random dataset of 40 samples
    dataset = []
    for _ in range(3000):
        robot_state = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1.0, 1.0),  # vx
            np.random.uniform(-1.0, 1.0)   # vy
        ]
        n_obs = np.random.randint(2, 5)
        obstacles = []
        for __ in range(n_obs):
            ox = np.random.uniform(0.5, 1.5)
            oy = np.random.uniform(0.5, 1.5)
            r  = np.random.uniform(0.1, 0.3)
            obstacles.append([ox, oy, r])
        goal = [1.5, 1.5]
        risk = np.random.uniform(0.0, 2.0)

        dataset.append({
            'robot': robot_state,
            'obstacles': obstacles,
            'goal': goal,
            'risk': risk
        })

    # Convert to graph dataset
    graphs_dataset = module.wrap_dataset_to_graphs(dataset)
    
    # Example: Print a sample graph
    print("Sample graph node features:\n", graphs_dataset[0].x)
    print("Sample graph edge index:\n", graphs_dataset[0].edge_index)
    print("Sample graph edge attributes:\n", graphs_dataset[0].edge_attr)
    print("Sample graph robot batch:\n", graphs_dataset[0].batch)
    print("Sample graph target (risk):\n", graphs_dataset[0].y)

    # Create a DataLoader for batching
    batching_data_loader = DataLoader(graphs_dataset, batch_size=2, shuffle=True)
    for sample_batch in batching_data_loader:
        sample_batch = sample_batch
        print("Batch node features (x):", sample_batch.x)
        print("Batch edge index:", sample_batch.edge_index)
        print("Batch edge attributes (edge_attr):", sample_batch.edge_attr)
        print("Batch target (y):", sample_batch.y)
        print("Batch mapping (sample_batch):", sample_batch.batch)
        break


    # Train/test split
    train_data, test_data = train_test_split(graphs_dataset, test_size=0.2, random_state=42)

    # Train
    module.train_model(train_data, test_data)

    # Evaluate
    metrics = module.evaluate_model(test_data)
    print("Test Metrics:", metrics)

    # Visualize an example graph
    module.visualize_graph(train_data[0])



if __name__ == "__main__":
    demo()
