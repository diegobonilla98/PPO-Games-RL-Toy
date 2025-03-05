import torch
import torch.nn as nn
import torch.nn.functional as F

class Brain(nn.Module):
    def __init__(self, in_channels, extra_dim, num_actions, emb_dim=64):
        """
        Simplified neural network for a PPO agent combining image and extra state inputs.

        Args:
            in_channels (int): Number of channels of the input image.
            extra_dim (int): Dimension of the extra state info vector.
            num_actions (int): Number of possible actions.
            emb_dim (int): Dimension of the embedding / hidden layer.
        """
        super(Brain, self).__init__()
        
        # Image branch: a simple CNN
        self.image_encoder = nn.Sequential(
            # For a 30x30 image: first conv reduces size to 15x15
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Second conv reduces size to 8x8
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten feature maps into a vector
            nn.Linear(32 * 8 * 8, emb_dim),
            nn.ReLU()
        )
        
        # Extra state branch: a simple MLP
        self.extra_encoder = nn.Sequential(
            nn.Linear(extra_dim, emb_dim),
            nn.ReLU()
        )
        
        # Mixer: combine image and extra state features
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU()
        )
        
        # Policy head: outputs logits for each action
        self.policy_head = nn.Linear(emb_dim, num_actions)
        # Value head: outputs a scalar state value
        self.value_head = nn.Linear(emb_dim, 1)
        
    def forward(self, image, extra):
        """
        Forward pass of the network.
        
        Args:
            image (torch.Tensor): Batch of images with shape [B, C, H, W].
            extra (torch.Tensor): Batch of extra state info with shape [B, extra_dim].
            
        Returns:
            policy_logits (torch.Tensor): Action logits [B, num_actions].
            value (torch.Tensor): State value estimates [B, 1].
        """
        img_emb = self.image_encoder(image)
        extra_emb = self.extra_encoder(extra)
        combined = torch.cat([img_emb, extra_emb], dim=1)
        hidden = self.fc(combined)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return policy_logits, value
    
    def get_action(self, image, extra):
        """
        Samples an action, and returns the action, its log probability, and the value.
        
        Args:
            image (torch.Tensor): Batch of images [B, C, H, W].
            extra (torch.Tensor): Batch of extra state info [B, extra_dim].
            
        Returns:
            action (torch.Tensor): Sampled action indices [B].
            log_prob (torch.Tensor): Log probabilities of the actions [B].
            value (torch.Tensor): State value estimates [B, 1].
        """
        policy_logits, value = self.forward(image, extra)
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

# Example usage:
if __name__ == "__main__":
    # For example, a 30x30 input image with 4 channels (if using a one-hot encoded state)
    dummy_image = torch.randn(1, 4, 30, 30)
    # Extra state could be any vector, here we use a 6-dimensional dummy vector
    dummy_extra = torch.randn(1, 6)

    model = Brain(in_channels=4, extra_dim=6, num_actions=5, emb_dim=128)
    policy_logits, value = model(dummy_image, dummy_extra)
    print("Policy logits:", policy_logits)
    print("Value estimate:", value)
