import os
import numpy as np
import torch
import torch.optim as optim
import copy
import tqdm
import FlappyBirdEnv
from flappy_brain import Brain


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


def get_difficulty(episode, min_eps=150, max_eps=500, decay_episodes=5_000):
    return 150
    return int(max(min_eps, max_eps - (max_eps - min_eps) * episode / decay_episodes))


def lerp(value, in_min, in_max, out_min, out_max):
    """Automatically determines interpolation direction based on input order."""
    if out_min > out_max:
        # Reverse the interpolation if output range is decreasing
        return out_min + (in_max - value) * (out_max - out_min) / (in_max - in_min)
    else:
        # Standard interpolation
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)



brain = Brain(num_inputs=5).cuda()
optimizer = optim.Adam(brain.parameters(), lr=1e-4)
old_brain = Brain(num_inputs=5).cuda()

# Load weights from file into both:
if os.path.exists("ppo_flappy.pth"):
    print("Loading weights from file...")
    checkpoint_path = "ppo_flappy.pth"
    state_dict = torch.load(checkpoint_path)
    brain.load_state_dict(state_dict)
    old_brain.load_state_dict(state_dict)

K = 500000  # episodes or iterations
eps_clip = 0.2
ent_coef = 0.01
vf_coef = 0.5
n_epochs = 10
batch_size = 128
max_episode_length = 4000

all_rewards = []

continuous_solved_episodes = 0

pbar = tqdm.tqdm(total=K, unit="episode")
try:
    for episode in range(K):
        pbar.update()
        # collect one episode (or set # of steps) of data
        D = []
        FlappyBirdEnv.reset()
        state = FlappyBirdEnv.get_state()
        FlappyBirdEnv.pipe_gap = get_difficulty(episode)
        done = False
        episode_reward = 0
        max_episode_length_weighted = lerp(get_difficulty(episode), 500, 150, 500, max_episode_length)
        while not done and len(D) < max_episode_length_weighted:
            s_tensor = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
            with torch.no_grad():
                action, logp, value = brain.get_action(s_tensor)
            action_int = int(action.item())
            next_r, _, done = FlappyBirdEnv.update(action_int)
            episode_reward += next_r

            D.append({
                's': state,
                'a': action_int,
                'logp': logp.item(),
                'v': value.item(),
                'r': next_r,
            })
            state = FlappyBirdEnv.get_state()  # or next state if your env returns it
            # FlappyBirdEnv.draw()

        # compute advantage + returns via GAE
        D = compute_gae_and_returns(D, gamma=0.95, lam=0.95)

        # copy old_brain
        old_brain.load_state_dict(copy.deepcopy(brain.state_dict()))

        # Now train for n_epochs with mini-batches
        # Flatten or convert D to Tensors
        states = torch.tensor([d['s'] for d in D], dtype=torch.float32, device="cuda")
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
                mb_states = states[mb_idxs]
                mb_actions = actions[mb_idxs]
                mb_old_log_probs = old_log_probs[mb_idxs]
                mb_returns = returns[mb_idxs]
                mb_advantages = advantages[mb_idxs]

                # forward pass in new policy
                policy_logits, v_pred = brain(mb_states)
                dist = torch.distributions.Bernoulli(logits=policy_logits)
                new_log_probs = dist.log_prob(mb_actions)

                # ratio
                ratio = (new_log_probs - mb_old_log_probs).exp()
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

        pbar.set_description(f"ep {episode}, r: {episode_reward}, gap: {FlappyBirdEnv.pipe_gap}, ep. length: {len(D)}")
        all_rewards.append(episode_reward)
        if episode > 1_000 and episode_reward > 100:
            continuous_solved_episodes += 1
        else:
            continuous_solved_episodes = 0
        if continuous_solved_episodes > 50:
            print("Solved!")
            break

        if episode > 0 and episode % 100 == 0:
            torch.save(brain.state_dict(), "ppo_flappy.pth")
            np.save("ppo_flappy_rewards.npy", all_rewards)

except KeyboardInterrupt:
    pass

torch.save(brain.state_dict(), "ppo_flappy.pth")
all_rewards = np.array(all_rewards)
np.save("ppo_flappy_rewards.npy", all_rewards)
