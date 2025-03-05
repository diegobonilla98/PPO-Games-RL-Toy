import pygame
import sys
import random

# Initialize Pygame
pygame.init()
is_main = False

# Screen settings
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Flappy Bird")
clock = pygame.time.Clock()

background_image = pygame.image.load("./background-day.png").convert()
bird_image = pygame.image.load("./yellowbird-downflap.png").convert_alpha()
pipe_image = pygame.image.load("./pipe-green.png").convert_alpha()

# If your background needs to fill the window, do:
screen_width, screen_height = screen.get_size()
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))

# Bird settings
bird_x = 50
bird_y = HEIGHT // 2
bird_width = 40
bird_height = 30
bird_velocity = 0
gravity = 0.5
jump_velocity = -10

# Pipe settings
pipe_width = 70
pipe_gap = 500  # 150
pipe_speed = 3
last_pipe_time = pygame.time.get_ticks()
pipes = []  # list to hold pairs of top and bottom pipes

episode_reward = 0
frame_count = 0
pipe_interval_steps = 90


class Pipe(pygame.Rect):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.passed = False  # Add custom attribute


def reset():
    global bird_y, bird_velocity, pipes, episode_reward, frame_count
    bird_y = HEIGHT // 2
    bird_velocity = 0
    episode_reward = 0
    frame_count = 0
    pipes = []


def add_pipe():
    """Generate a new pair of pipes with a random gap position."""
    # Random height for the top pipe (minimum height 50 pixels)
    top_height = random.randint(50, HEIGHT - pipe_gap - 50)

    # Create pipe objects using the custom Pipe class
    top_pipe = Pipe(WIDTH, 0, pipe_width, top_height)
    bottom_pipe = Pipe(WIDTH, top_height + pipe_gap, pipe_width, HEIGHT - top_height - pipe_gap)

    pipes.append((top_pipe, bottom_pipe))


def update(action):
    global bird_velocity, bird_y, last_pipe_time, pipes, episode_reward, running, bird_rect, frame_count

    frame_count += 1
    
    inmediate_reward = 0.1
    episode_reward += 0.1  # 0.005
    crashed = False

    if action == 1:
        bird_velocity = jump_velocity

    # Update bird physics
    bird_velocity += gravity
    bird_y += bird_velocity

    # Generate new pipes periodically
    if len(pipes) == 0 or frame_count % pipe_interval_steps == 0:
        add_pipe()

    # Move pipes to the left and remove those off-screen
    for pipe_pair in pipes:
        top_pipe, bottom_pipe = pipe_pair
        top_pipe.x -= pipe_speed
        bottom_pipe.x -= pipe_speed
    pipes = [pair for pair in pipes if pair[0].right > 0]

    # Create bird rectangle for collision detection
    bird_rect = pygame.Rect(bird_x, bird_y, bird_width, bird_height)

    # Check for collisions with the screen bounds
    if bird_y < 0 or bird_y + bird_height > HEIGHT:
        episode_reward -= 3.0
        inmediate_reward = -3.0
        crashed = True
        if is_main:
            print(episode_reward)
            reset()
            running = False

    # Check for collisions with pipes
    for top_pipe, bottom_pipe in pipes:
        if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
            episode_reward -= 3.0
            inmediate_reward = -3.0
            crashed = True
            if is_main:
                print(episode_reward)
                reset()
                running = False

    for top_pipe, bottom_pipe in pipes:
        if bird_x > top_pipe.x + pipe_width and not top_pipe.passed:
            episode_reward += 1.0  # Give reward for passing a pipe
            inmediate_reward = 1.0
            top_pipe.passed = True
    if is_main:
        s = get_state()
        print(s)
    return inmediate_reward, episode_reward, crashed


def get_state():
    global pipes
    closest_pipe = None
    for pipe_pair in pipes:
        if pipe_pair[0].x > bird_x:
            closest_pipe = pipe_pair
            break
    if closest_pipe:
        gap_center_y = closest_pipe[0].height + (pipe_gap / 2)
        bird_to_gap = (bird_y - gap_center_y) / HEIGHT
        return bird_to_gap, bird_y / HEIGHT, bird_velocity, closest_pipe[0].x / WIDTH, pipe_gap / HEIGHT
    else:
        return 1.0, bird_y / HEIGHT, bird_velocity, 1.0, pipe_gap / HEIGHT  # Default values when no pipes are ahead


def sad_draw():
    global screen, pipes
    screen.fill((135, 206, 235))  # fill with sky blue
    pygame.draw.rect(screen, (255, 255, 0), bird_rect)  # draw the bird in yellow
    for top_pipe, bottom_pipe in pipes:
        pygame.draw.rect(screen, (0, 255, 0), top_pipe)  # draw pipes in green
        pygame.draw.rect(screen, (0, 255, 0), bottom_pipe)
    pygame.display.flip()


def draw():
    global screen, bird_rect, pipes

    # 1) Draw background (already scaled to screen size once)
    screen.blit(background_image, (0, 0))
    
    # 2) Draw the bird, scaled to the collision rectangle
    resized_bird = pygame.transform.scale(bird_image, (bird_rect.width, bird_rect.height))
    screen.blit(resized_bird, (bird_rect.x, bird_rect.y))
    
    # 3) Prepare pipe images (width=pipe_width, keep aspect ratio)
    #    We'll do this once here, then re-use for each top/bottom pipe.
    pipe_ratio = pipe_image.get_height() / pipe_image.get_width()
    scaled_pipe_height = int(pipe_width * pipe_ratio)
    
    #    Normal (bottom) pipe
    pipe_bottom_img = pygame.transform.scale(pipe_image, (pipe_width, scaled_pipe_height))
    #    Flipped (top) pipe
    pipe_top_img = pygame.transform.flip(pipe_bottom_img, False, True)
    
    for top_pipe, bottom_pipe in pipes:
        #
        # --- Top pipe ---
        #
        # We only want to show the *bottom* 'top_pipe.height' portion of the flipped image.
        # The top pipe has top_pipe.y = 0 and top_pipe.height = how tall it is.
        # We'll clip from the bottom of the flipped image (pipe_top_img).
        if top_pipe.height > 0:  # just a safety check
            top_clip_rect = pygame.Rect(
                0,                             # left offset in the image
                pipe_top_img.get_height() - top_pipe.height,  # top offset in the image
                pipe_top_img.get_width(), 
                top_pipe.height
            )
            # Blit at (top_pipe.x, 0), so it lines up exactly with the collision rect
            # If top_pipe.height is bigger than pipe_top_img, it will clip automatically.
            screen.blit(pipe_top_img, (top_pipe.x, top_pipe.y), top_clip_rect)

        #
        # --- Bottom pipe ---
        #
        # We only want the *top* 'bottom_pipe.height' portion of the normal image.
        if bottom_pipe.height > 0:  # safety check
            bottom_clip_rect = pygame.Rect(
                0,             # left offset in the image
                0,             # top offset in the image (we want from the top down)
                pipe_bottom_img.get_width(),
                bottom_pipe.height
            )
            screen.blit(pipe_bottom_img, (bottom_pipe.x, bottom_pipe.y), bottom_clip_rect)

    return screen


if __name__ == '__main__':
    # running = True
    is_main = True
    k = 0
    while True:
        action = 0
        clock.tick(30)

        k += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(k)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1

        update(action)
        screen = draw()
        pygame.display.flip()
