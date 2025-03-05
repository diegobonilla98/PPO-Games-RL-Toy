import torch
import torch.nn as nn
import torch.nn.functional as F


class Brain(nn.Module):
    def __init__(self, num_actions=5, extra_features_dim=5):
        """
        Args:
            num_actions (int): Number of possible actions (0: up, 1: right, 2: down, 3: left).
            extra_features_dim (int): Dimension of extra 1D features. Here we use 5:
                - (snake_x, snake_y, food_x, food_y, distance).
        """
        super(Brain, self).__init__()
        
        # Convolutional layers for RGB frame processing
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for CNN features + extra features
        self.fc1 = nn.Linear(32 + extra_features_dim, 128)
        
        # Actor (policy) and critic (value) heads
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, image, extra):
        """
        Forward pass of the network.

        Args:
            image (torch.Tensor): RGB images of shape [B, 3, H, W].
            extra (torch.Tensor): Extra 1D features of shape [B, extra_features_dim].
                Contains (snake_x, snake_y, food_x, food_y, distance).
                
        Returns:
            policy_logits (torch.Tensor): Logits for each action, shape [B, num_actions].
            value (torch.Tensor): Value estimates, shape [B, 1].
        """
        # Process image through convolutional layers
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten CNN output, shape [B, 32]
        
        # Concatenate extra features
        x = torch.cat([x, extra], dim=1)  # Shape [B, 32 + extra_features_dim]
        x = F.relu(self.fc1(x))
        
        # Actor and Critic outputs
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def get_action(self, image, extra):
        """
        Returns sampled action, log_prob, and value for the current state.

        Args:
            image (torch.Tensor): RGB images of shape [B, 3, H, W].
            extra (torch.Tensor): Extra 1D features of shape [B, extra_features_dim].

        Returns:
            action (torch.Tensor): Sampled action index, shape [B].
            log_prob (torch.Tensor): Log probability of the chosen action, shape [B].
            value (torch.Tensor): State value estimate, shape [B, 1].
        """
        policy_logits, value = self.forward(image, extra)
        
        # For multiple discrete actions, we use a Categorical distribution
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()          # Sample action
        log_prob = dist.log_prob(action) # Log probability of that action
        
        return action, log_prob, value


# Example usage:
if __name__ == "__main__":
    dummy_image = torch.randn(1, 3, 64, 64)  # 1 sample, RGB image
    dummy_extra = torch.tensor([[0.5, 0.5, 0.8, 0.2, 0.6, 1.0]])  # Normalized (x, y, x_food, y_food, distance)

    model = Brain(num_actions=5, extra_features_dim=6)
    policy_logits, value = model(dummy_image, dummy_extra)
    print("Policy logits:", policy_logits)
    print("Value estimate:", value)
