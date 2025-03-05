import copy
import os
import tqdm
from SnakeEnv import SnakeGame, CELL_COUNT, pygame
from much_better_snake_brain import SnakeBrain
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.transforms import InterpolationMode


def compute_gae_and_returns(rollout, gamma=0.99, lam=0.95):
    """
    rollout is a list of dict or tuple: (s, a, r, v, logp)
    We'll add 'adv' and 'ret' to each.
    """
    advantages = []
    gae = 0
    # we assume rollout has an extra 'value' at the end for s_{T+1} or we do 0
    # or handle done if it's a true episode
    for t in reversed(range(len(rollout))):
        r = rollout[t]['r']
        v = rollout[t]['v']
        v_next = rollout[t+1]['v'] if t+1 < len(rollout) else 0.0  # if done
        delta = r + gamma * v_next - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    # Now compute returns
    for t in range(len(rollout)):
        rollout[t]['adv'] = advantages[t]
        rollout[t]['ret'] = rollout[t]['v'] + advantages[t]  # V + A
    return rollout


def get_difficulty(step, min_val=1, max_val=10, total_steps=5000):
    return int(max(min_val, max_val - (max_val - min_val) * step / total_steps))


def lerp(value, in_min, in_max, out_min, out_max):
    """Automatically determines interpolation direction based on input order."""
    if out_min > out_max:
        # Reverse the interpolation if output range is decreasing
        return out_min + (in_max - value) * (out_max - out_min) / (in_max - in_min)
    else:
        # Standard interpolation
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def get_no_progress_threshold(episode, difficulty):
    """
    Returns how many consecutive 'no-food' steps to allow before ending the
    episode early, based on current episode index and difficulty level.
    """
    # For example, from 100 steps at the start up to 10,000 steps after 50k episodes.
    # Also add a small bonus for higher difficulty, so that difficulty=10 allows more steps.
    
    min_val = 100       # Starting threshold for no-progress steps
    max_val = 10000      # Maximum threshold after enough training
    total_episodes = 50000  # After this many episodes, we reach max_val
    
    # Bonus for higher difficulty: up to ~1800 more steps (for difficulty=10)
    difficulty_bonus = (difficulty - 1) * 200
    
    # Clamp episode so it doesn’t exceed total_episodes
    e = min(episode, total_episodes)
    fraction = e / total_episodes  # 0.0 -> 1.0 as we go from ep=0 -> ep=50k
    
    # Interpolate linearly: 1,000 -> 10,000
    scaled_portion = min_val + fraction * (max_val - min_val)
    
    return int(scaled_portion + difficulty_bonus)


def states_to_tensors(states):
    # Convert states to tensors
    snake_bodies = [s['snake_body'] for s in states]
    head_locations = [s['head_location'] for s in states]
    snake_lengths = [s['snake_length'] for s in states]
    fruits = [s['fruits'] for s in states]

    snake_bodies = [[(sb[0] / CELL_COUNT, sb[1] / CELL_COUNT) for sb in snake_body] for snake_body in snake_bodies]
    head_locations = [(head_location[0] / CELL_COUNT, head_location[1] / CELL_COUNT) for head_location in head_locations]
    snake_lengths = [[sl / CELL_COUNT] for sl in snake_lengths]
    fruits = [[(f[0] / CELL_COUNT, f[1] / CELL_COUNT) for f in fruit] for fruit in fruits]

    snake_bodies = torch.tensor(snake_bodies, dtype=torch.float32, device="cuda")
    head_locations = torch.tensor(head_locations, dtype=torch.float32, device="cuda")
    snake_lengths = torch.tensor(snake_lengths, dtype=torch.float32, device="cuda")
    fruits = torch.tensor(fruits, dtype=torch.float32, device="cuda")

    return snake_bodies, head_locations, snake_lengths, fruits


snake_game = SnakeGame()

extra_dim = snake_game.get_extra_dim()
brain_args = {
    "n_fruits": snake_game.difficulty,  # we used 2 fruits here
    "snake_hidden_dim": 32,
    "body_output_dim": 32,
    "aux_hidden_dim": 32,
    "final_hidden_dim": 64
}
brain = SnakeBrain(**brain_args).cuda()
old_brain = SnakeBrain(**brain_args).cuda()

optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)

# Load weights from file into both:
# if os.path.exists("ppo_simple_snake.pth"):
#     ans = input("Load weights from file? (y/n): ")
#     if ans.lower() == "y":
#         print("Loading weights from file...")
#         checkpoint_path = "ppo_simple_snake.pth"
#         state_dict = torch.load(checkpoint_path)
#         brain.load_state_dict(state_dict)
#         old_brain.load_state_dict(state_dict)
#         if os.path.exists("ppo_simple_snake_optim.pth"):
#             print("Loading optimizer state...")
#             optimizer.load_state_dict(torch.load("ppo_simple_snake_optim.pth"))
        

K = 500000  # episodes or iterations
eps_clip = 0.2
ent_coef = 0.002
vf_coef = 0.5
n_epochs = 5
batch_size = 128
max_episode_length = 10000

all_rewards = []

pbar = tqdm.tqdm(total=K, unit="episode")
try:
    for episode in range(K):
        pbar.update()
        # collect one episode (or set # of steps) of data
        D = []

        # The maximum consecutive no-food steps allowed for this episode:
        max_no_progress = get_no_progress_threshold(episode, snake_game.difficulty)
        steps_without_food = 0

        snake_game.reset()
        state = snake_game.get_state_simple()
        done = False
        episode_reward = 0
        while not done:
            snake_body, head_location, snake_length, fruits = states_to_tensors([state])
            with torch.no_grad():
                action, logp, value = brain.get_action(snake_body, head_location, snake_length, fruits)
            action_int = int(action.item())
            next_r, done = snake_game.step(action_int)

            episode_reward += next_r
                
            D.append({
                'a': action_int,
                'logp': logp.item(),
                'v': value.item(),
                'r': next_r,
                's': state
            })
            state = snake_game.get_state_simple()
            snake_game.draw()
            pygame.display.flip()

        # compute advantage + returns via GAE
        D = compute_gae_and_returns(D, gamma=0.99, lam=0.95)

        # copy old_brain
        old_brain.load_state_dict(copy.deepcopy(brain.state_dict()))

        # Now train for n_epochs with mini-batches
        # Flatten or convert D to Tensors
        states = [d['s'] for d in D]
        actions = torch.tensor([d['a'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        old_log_probs = torch.tensor([d['logp'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        returns = torch.tensor([d['ret'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        advantages = torch.tensor([d['adv'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)

        # optional: normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(n_epochs):
            # you can shuffle data and do multiple mini-batches
            idxs = torch.randperm(len(D))
            for start in range(0, len(D), batch_size):
                end = start + batch_size
                mb_idxs = idxs[start:end]
                mb_states = [states[i] for i in mb_idxs]
                mb_states = states_to_tensors(mb_states)
                mb_actions = actions[mb_idxs]
                mb_old_log_probs = old_log_probs[mb_idxs]
                mb_returns = returns[mb_idxs]
                mb_advantages = advantages[mb_idxs]

                # forward pass in new policy
                policy_logits, v_pred = brain(*mb_states)

                # ratio
                ratio = (policy_logits.gather(1, mb_actions.long()) - mb_old_log_probs).exp()
                dist = torch.distributions.Categorical(logits=policy_logits)
                # clipped loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = (mb_returns - v_pred).pow(2).mean()

                # entropy
                entropy = dist.entropy().mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pbar.set_description(f"ep {episode}, r: {episode_reward}, difficulty: {snake_game.difficulty}, length: {len(D)}")
        all_rewards.append(episode_reward)
        if episode_reward > 1000:
            print("Solved!")
            break

        if episode > 0 and episode % 100 == 0:
            torch.save(brain.state_dict(), "ppo_simple_snake.pth")
            np.save("ppo_snake_simple_rewards.npy", all_rewards)
            torch.save(optimizer.state_dict(), "ppo_simple_snake_optim.pth")

except KeyboardInterrupt:
    pass

torch.save(brain.state_dict(), "ppo_simple_snake.pth")
all_rewards = np.array(all_rewards)
np.save("ppo_snake_simple_rewards.npy", all_rewards)
