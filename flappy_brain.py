import torch
import torch.nn as nn


class Brain(nn.Module):
    def __init__(self, num_inputs, dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(num_inputs, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(dim, 1)  # for Bernoulli
        self.value_head = nn.Linear(dim, 1)

    def forward(self, x):
        feat = self.features(x)
        policy_logit = self.policy_head(feat)
        value = self.value_head(feat)
        return policy_logit, value

    def get_action(self, state):
        """
        Returns sampled action, log_prob, and value for state.
        """
        policy_logit, value = self.forward(state)
        dist = torch.distributions.Bernoulli(logits=policy_logit)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
