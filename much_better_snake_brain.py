import math
import torch
from torch_geometric.data import Data, Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def snake2graph(list_of_snakes):
    """
    Converts a batch of snake bodies into a batched graph for PyTorch Geometric.
    Here we treat the snake as an undirected chain graph (node i connected to i+1).
    
    Args:
        list_of_snakes (List[List[Tuple[int,int]]]):
            Each element in the list is a snake body of variable length.
            snake[0] = head, snake[-1] = tail.
    
    Returns:
        Batch: A PyG Batch object containing:
               - x (Tensor): Node features of shape [total_nodes, 2] (x,y coords).
               - edge_index (Tensor): Graph edges of shape [2, total_edges].
               - batch (Tensor): Batch assignment for each node.
    """
    data_list = []
    
    for snake in list_of_snakes:
        # Node features: (x, y)
        node_feats = []
        edge_index_list = []

        n_nodes = len(snake)
        for (x_coord, y_coord) in snake:
            node_feats.append([x_coord, y_coord])
        
        # Build edges
        for i in range(n_nodes - 1):
            # i -> i+1
            edge_index_list.append([i, i+1])
        
        node_feats = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()  # shape [2, E]
        
        data = Data(x=node_feats, edge_index=edge_index)
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list).cuda()  # Move to GPU if desired
    return batch


class SnakeBodyEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=32):
        """
        A simpler 2-layer GNN for encoding the snake body as a chain.
        
        Args:
            input_dim  (int): Node feature dimension (x,y) => 2.
            hidden_dim (int): Hidden dimension for GNN layers.
            output_dim (int): Final embedding size for the entire snake body.
        """
        super().__init__()
        # Two GCNConv layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Final linear layer to produce the output embedding
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        """
        Args:
            batch (Batch): PyG Batch object from snake2graph().
        
        Returns:
            Tensor of shape [batch_size, output_dim] 
            representing the entire snake body embedding.
        """
        # GCNConv layer 1
        h = self.conv1(batch.x, batch.edge_index)
        h = F.relu(h)
        
        # GCNConv layer 2
        h = self.conv2(h, batch.edge_index)
        h = F.relu(h)
        
        # Pool (mean) over each separate snake in the batch
        h_pool = global_mean_pool(h, batch.batch)
        
        # Final readout
        out = self.readout(h_pool)
        return out


class SnakeBrain(nn.Module):
    def __init__(self,
                 n_fruits=1,
                 snake_hidden_dim=32,
                 body_output_dim=32,
                 aux_hidden_dim=32,
                 final_hidden_dim=64):
        """
        A combined network that:
         - Encodes snake body via a simpler GCN
         - MLP-encodes head location, body size, fruits
         - Produces (logits for 5 actions) and (scalar value).

        Args:
            n_fruits         (int): Number of fruits (constant).
            snake_hidden_dim (int): Hidden dim in GNN layers.
            body_output_dim  (int): Final embedding size from GNN.
            aux_hidden_dim   (int): Hidden dim for processing the other features.
            final_hidden_dim (int): Hidden dim for the combined features 
                                    (body_emb + aux_emb).
        """
        super().__init__()
        
        # GNN to encode snake body (chain)
        self.body_encoder = SnakeBodyEncoder(
            input_dim=2,             # (x, y)
            hidden_dim=snake_hidden_dim,
            output_dim=body_output_dim
        )
        
        # We'll encode the rest of the features (head loc, body size, fruit loc)
        # head: 2 dims
        # body_size: 1 dim
        # fruits: n_fruits * 2
        self.aux_in_dim = 2 + 1 + (n_fruits * 2)
        
        self.aux_mlp = nn.Sequential(
            nn.Linear(self.aux_in_dim, aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(aux_hidden_dim, aux_hidden_dim),
            nn.ReLU()
        )
        
        # Combine the GNN output and the aux embedding
        self.combined_mlp = nn.Sequential(
            nn.Linear(body_output_dim + aux_hidden_dim, final_hidden_dim),
            nn.ReLU()
        )
        
        # Policy head: 5 discrete actions => 5 logits
        self.policy_head = nn.Linear(final_hidden_dim, 5)
        # Value head: single scalar
        self.value_head = nn.Linear(final_hidden_dim, 1)

    def forward(self, list_of_snakes, heads, body_sizes, fruits):
        """
        Args:
            list_of_snakes (list): 
                A list (of length B) of variable-length snake bodies 
                e.g. [ [(x0,y0),(x1,y1),...],  ..., [...] ]
            heads (Tensor): shape [B, 2], the (x, y) head location 
            body_sizes (Tensor): shape [B, 1], the length of the snake or similar 
            fruits (Tensor): shape [B, N_FRUITS, 2]
        
        Returns:
            (logits, value)
            logits: Tensor of shape [B, 5]
            value:  Tensor of shape [B, 1]
        """
        # 1) Encode snake body with simpler GCN
        batch = snake2graph(list_of_snakes)
        body_emb = self.body_encoder(batch)  # shape [B, body_output_dim]
        
        # 2) Encode the auxiliary features
        B, n_fruits, _ = fruits.shape
        fruits_flat = fruits.view(B, -1)  # [B, 2*N_FRUITS]
        aux_input = torch.cat([heads, body_sizes, fruits_flat], dim=-1)
        aux_emb = self.aux_mlp(aux_input)  # [B, aux_hidden_dim]
        
        # 3) Combine GNN embedding + aux embedding
        combined = torch.cat([body_emb, aux_emb], dim=-1)
        combined = self.combined_mlp(combined)  # [B, final_hidden_dim]
        
        # 4) Produce policy logits and value
        logits = self.policy_head(combined)  # [B, 5]
        value = self.value_head(combined)    # [B, 1]
        
        return logits, value

    def get_action(self, list_of_snakes, heads, body_sizes, fruits):
        logits, value = self.forward(list_of_snakes, heads, body_sizes, fruits)
        
        # For multiple discrete actions, we use a Categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()          # Sample action
        log_prob = dist.log_prob(action) # Log probability of that action
        
        return action, log_prob, value


if __name__ == "__main__":
    # Example usage
    snake_batch = [
        [(5,5),(5,4),(5,3)],
        [(10,10),(10,9),(9,9),(8,9)]
    ]
    
    # Suppose we have 2 fruits for each state
    fruits = torch.tensor([
        [[3,3],[7,7]],
        [[15,1],[5,10]]
    ], dtype=torch.float).cuda()

    heads = torch.tensor([[5,5],
                          [10,10]], dtype=torch.float).cuda()
    body_sizes = torch.tensor([[3],
                               [4]], dtype=torch.float).cuda()
    
    model = SnakeBrain(n_fruits=2,
                       snake_hidden_dim=32,
                       body_output_dim=32,
                       aux_hidden_dim=32,
                       final_hidden_dim=64).cuda()
    
    logits, value = model(snake_batch, heads, body_sizes, fruits)
    print("logits shape =", logits.shape)  # [2, 5]
    print("value shape  =", value.shape)   # [2, 1]
