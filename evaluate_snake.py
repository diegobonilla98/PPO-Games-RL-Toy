import torch
import pygame
import numpy as np
from SnakeEnv import SnakeGame
# from snake_brain import Brain
from better_snake_brain import Brain
import imageio
import datetime


# Initialize the game environment
snake_game = SnakeGame()
# snake_game.difficulty = 5

# Load the trained model
brain_args = {
    'in_channels': 4,
    'extra_dim': 8,
    'num_actions': 5,
    'emb_dim': 128
}
brain = Brain(**brain_args).cuda()
brain.load_state_dict(torch.load("ppo_snake.pth"))
brain.eval()

# Game settings
FPS = 30  # Adjust as needed for smoothness

save_replay = True

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

frames = []

font = pygame.font.SysFont(None, 35)

snake_game.reset()
done = False
score = 0
total_frames = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            done = True
            break

    total_frames += 1

    # Get the current state
    image, extra = snake_game.get_state()
    image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda").unsqueeze(0)
    extra_tensor = torch.tensor(extra, dtype=torch.float32, device="cuda").unsqueeze(0)

    # Select action
    with torch.no_grad():
        action, _, value = brain.get_action(image_tensor, extra_tensor)
    action_int = int(action.item())

    # Take a step in the environment
    reward, done = snake_game.step(action_int)
    if reward > snake_game.food_reward * 0.9:
        score += 1

    # Render the game
    snake_game.draw()

    # Print score in game
    score_text = font.render(f'Score: {score}', True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    # Print immediate reward in game
    reward_text = font.render(f'Reward: {round(reward, 2)}', True, (255, 255, 255))
    screen.blit(reward_text, (10, 40))

    # Print value in game
    value_text = font.render(f'Value: {round(value.item(), 2)}', True, (255, 255, 255))
    screen.blit(value_text, (10, 70))

    pygame.display.flip()
    clock.tick(FPS)

    if save_replay:
        # Save the frame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Convert to (height, width, channels)
        frames.append(frame)

    if done:
        break

print(f"Final score: {score}")
print(f"Total frames: {total_frames}")
# Save the frames as a video
if save_replay:
    imageio.mimsave("snake_gameplay_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp4", frames, fps=60)
pygame.quit()
