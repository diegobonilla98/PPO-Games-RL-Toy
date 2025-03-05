import copy
import os
import tqdm
from SnakeEnv import SnakeGame, pygame
# from snake_brain import Brain
from better_snake_brain import Brain
import torch
import numpy as np


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


def get_max_episode_length(current_episode: int, total_episodes: int = 100_000, init_length: int = 500, final_length: int = 10_000, warmup_fraction: float = 0.2):
    warmup_episodes = int(total_episodes * warmup_fraction)
    
    if current_episode < warmup_episodes:
        # Linear interpolation from init_length to final_length
        ratio = current_episode / warmup_episodes
        max_length = init_length + ratio * (final_length - init_length)
        return int(max_length)
    else:
        return final_length


snake_game = SnakeGame()

brain_args = {
    'in_channels': 4,
    'extra_dim': 8,
    'num_actions': 5,
    'emb_dim': 128
}
brain = Brain(**brain_args).cuda()
old_brain = Brain(**brain_args).cuda()

optimizer = torch.optim.Adam(brain.parameters(), lr=1e-4)

display = False

# Load weights from file into both:
if not display and os.path.exists("ppo_snake.pth"):
    ans = input("Load weights from file? (y/n): ")
    if ans.lower() == "y":
        print("Loading weights from file...")
        checkpoint_path = "ppo_snake.pth"
        state_dict = torch.load(checkpoint_path)
        brain.load_state_dict(state_dict)
        old_brain.load_state_dict(state_dict)
        if os.path.exists("ppo_snake_optim.pth"):
            optimizer.load_state_dict(torch.load("ppo_snake_optim.pth"))
            print("Loaded optimizer state.")

K = 200_000  # episodes or iterations
eps_clip = 0.1
ent_coef = 0.02
vf_coef = 0.5
n_epochs = 10
batch_size = 256

all_rewards = []
all_losses = []
very_good_episodes = 0

pbar = tqdm.tqdm(total=K, unit="episode")
try:
    for episode in range(K):
        pbar.update()
        # collect one episode (or set # of steps) of data
        D = []

        current_max_length = get_max_episode_length(current_episode=episode, total_episodes=K, init_length=500, final_length=10_000, warmup_fraction=0.2)

        snake_game.reset()
        image, extra = snake_game.get_state()
        done = False
        episode_reward = 0
        while not done:
            image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda").unsqueeze(0)
            extra_tensor = torch.tensor(extra, dtype=torch.float32, device="cuda").unsqueeze(0)

            with torch.no_grad():
                action, logp, value = brain.get_action(image_tensor, extra_tensor)
            action_int = int(action.item())
            next_r, done = snake_game.step(action_int)
            
            episode_reward += next_r
                
            D.append({
                'a': action_int,
                'logp': logp.item(),
                'v': value.item(),
                'r': next_r,
                's': (image, extra)
            })
            image, extra = snake_game.get_state()

            if len(D) >= current_max_length:
                break
            
            if display:
                snake_game.draw()
                pygame.display.flip()

        # compute advantage + returns via GAE
        D = compute_gae_and_returns(D, gamma=0.99, lam=0.95)

        # copy old_brain
        old_brain.load_state_dict(copy.deepcopy(brain.state_dict()))

        # Now train for n_epochs with mini-batches
        # Flatten or convert D to Tensors
        states = torch.tensor([d['s'][0] for d in D], dtype=torch.float32, device="cuda")
        extra = torch.tensor([d['s'][1] for d in D], dtype=torch.float32, device="cuda")
        actions = torch.tensor([d['a'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        old_log_probs = torch.tensor([d['logp'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        returns = torch.tensor([d['ret'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)
        advantages = torch.tensor([d['adv'] for d in D], dtype=torch.float32, device="cuda").unsqueeze(-1)

        # optional: normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ep_loss = 0
        for epoch in range(n_epochs):
            # you can shuffle data and do multiple mini-batches
            idxs = torch.randperm(len(D))
            for start in range(0, len(D), batch_size):
                end = start + batch_size
                mb_idxs = idxs[start:end]
                mb_states = states[mb_idxs]
                mb_extra = extra[mb_idxs]
                mb_actions = actions[mb_idxs]
                mb_old_log_probs = old_log_probs[mb_idxs]
                mb_returns = returns[mb_idxs]
                mb_advantages = advantages[mb_idxs]

                # forward pass in new policy
                policy_logits, v_pred = brain(mb_states, mb_extra)

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
                ep_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_losses.append(ep_loss / n_epochs)
        pbar.set_description(f"ep {episode}, r: {episode_reward}, length: {len(D)}, max_length: {current_max_length}, loss: {all_losses[-1]:.2f}")
        all_rewards.append(episode_reward)
        if episode_reward > 3000:
            very_good_episodes += 1
            if very_good_episodes >= 50:
                break
        else:
            very_good_episodes = 0
        
        if episode > 0 and episode % 100 == 0:
            torch.save(brain.state_dict(), "ppo_snake.pth")
            np.save("ppo_snake_rewards.npy", all_rewards)
            np.save("ppo_snake_losses.npy", all_losses)
            torch.save(optimizer.state_dict(), "ppo_snake_optim.pth")
            print("Saved model and optimizer state.")
            
except KeyboardInterrupt:
    pass

torch.save(brain.state_dict(), "ppo_snake.pth")
all_rewards = np.array(all_rewards)
np.save("ppo_snake_rewards.npy", all_rewards)
np.save("ppo_snake_losses.npy", all_losses)
