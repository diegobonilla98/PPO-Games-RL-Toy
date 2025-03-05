import numpy as np
import torch
import pygame
import imageio
import FlappyBirdEnv
from FlappyBirdEnv import update, get_state, draw, reset, clock

# Load the trained model
class Brain(torch.nn.Module):
    def __init__(self, num_inputs, dim=64):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU()
        )
        self.policy_head = torch.nn.Linear(dim, 1)  # for Bernoulli
        self.value_head = torch.nn.Linear(dim, 1)

    def forward(self, x):
        feat = self.features(x)
        policy_logit = self.policy_head(feat)
        value = self.value_head(feat)
        return policy_logit, value

    def get_action_with_saliency(self, state):
        """
        Returns:
            action_probability (float),
            value (float),
            saliency (ndarray of shape [num_inputs]),
            state_for_display (ndarray of shape [num_inputs])
        """
        state_for_grad = state.clone().detach().requires_grad_(True)
        
        policy_logit, value = self.forward(state_for_grad)
        dist = torch.distributions.Bernoulli(logits=policy_logit)
        action_prob = dist.probs.item()  # Convert to probability

        # Backprop to get feature importance
        policy_logit.backward(retain_graph=True)
        saliency = state_for_grad.grad.abs().squeeze(0).detach().cpu().numpy()

        value_val = value.detach().cpu().item()
        state_for_display = state.squeeze(0).detach().cpu().numpy()
        
        return action_prob, value_val, saliency, state_for_display


# --- Main Script ---

# Input feature names
input_names = ["gap_distance", "bird_y", "bird_vel", "pipe_x", "pipe_gap"]

# Initialize agent
num_inputs = 5
brain = Brain(num_inputs=num_inputs).cuda()
brain.load_state_dict(torch.load("./ppo_flappy.pth"))
brain.eval()

FlappyBirdEnv.pipe_gap = 160

frames = []

# Font for on-screen text
pygame.font.init()
font = pygame.font.Font(None, 24)

reset()
done = False
total_reward = 0
score = 0

while True:
    clock.tick(0)  # Limit frame rate (0 = no limit)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    if done:
        break

    # Get current state
    raw_state = get_state()
    state = torch.tensor(raw_state, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Get action probability, value estimate, and saliency
    action_prob, value_val, saliency_vals, display_state = brain.get_action_with_saliency(state)

    # Step environment
    action = 1 if action_prob >= 0.5 else 0
    reward, _, done = update(action)
    if reward == 1.0:
        score += 1
    total_reward += reward

    # Draw the environment
    screen = draw()

    # Overlay debug info
    y_offset = 5

    # Display raw input values with names
    text_surface = font.render("Model Inputs:", True, (255, 255, 255))
    screen.blit(text_surface, (10, y_offset))
    y_offset += 20
    for i, (name, val) in enumerate(zip(input_names, display_state)):
        txt = f"{name}: {val:.3f}"
        text_surface = font.render(txt, True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 18

    y_offset += 5

    # Display action probability with color coding
    action_color = (0, 0, 255) if action_prob >= 0.5 else (255, 0, 0)  # Blue if >=50%, Red otherwise
    action_text = f"Flap Probability: {action_prob:.2%}"
    text_surface = font.render(action_text, True, action_color)
    screen.blit(text_surface, (10, y_offset))
    y_offset += 20

    # Display value estimate
    text_surface = font.render(f"Value (V): {value_val:.3f}", True, (255, 255, 255))
    screen.blit(text_surface, (10, y_offset))
    y_offset += 20

    # Display saliency per input with labels
    text_surface = font.render("Feature Importance:", True, (255, 255, 255))
    screen.blit(text_surface, (10, y_offset))
    y_offset += 20

    sal_scale = 50.0
    for i, (name, sval) in enumerate(zip(input_names, saliency_vals)):
        bar_length = min(int(sval * sal_scale), 200)
        pygame.draw.rect(screen, (255, 0, 0), (10, y_offset + i*15, bar_length, 10))
        txt = f"{name}: {sval:.4f}"
        text_surface = font.render(txt, True, (255, 255, 255))
        screen.blit(text_surface, (220, y_offset + i*15))

    y_offset += 15 * len(saliency_vals) + 5

    # Move score to bottom of screen
    y_offset = screen.get_height() - 30
    text_surface = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text_surface, (10, y_offset))

    pygame.display.flip()

    # Save the frame
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = np.transpose(frame, (1, 0, 2))
    frames.append(frame)

    if done:
        break

# Save the frames as a video
imageio.mimsave("flappy_gameplay_much_better.mp4", frames, fps=60)
print(f"Game Over! Total Reward: {total_reward}")
pygame.quit()
