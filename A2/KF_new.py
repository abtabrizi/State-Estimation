import pygame
import numpy as np
import imageio.v2 as imageio
import os

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1200, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

bot_img = pygame.image.load('car_128.png').convert_alpha()
car_w = 80
car_h = 30
bot_img = pygame.transform.scale(bot_img, (car_w, car_h))

# System parameters
T = 1/8  # Time step
r = 0.1  # Wheel radius
F = np.array([[1, 0], [0, 1]])  # State transition matrix
G = np.array([[r / 2 * T, r / 2 * T], [r / 2 * T, r / 2 * T]])  # Control matrix
Q = np.array([[0.1, 0], [0, 0.15]])  # Process noise covariance matrix
R = np.array([[0.05, 0], [0, 0.075]])  # Measurement noise covariance matrix
H = np.array([[1, 0], [0, 2]])  # Measurement matrix

x = np.array([[0], [0]])  # Initial state [x, y]
P = np.array([[0, 0], [0, 0]])  # Initial covariance matrix

frame_images = [] #List to store frame image filenames
frame_count = 0

ground_truth_positions = []
measured_positions = []

# Kalman Filter function
def kalman_filter(x, P, z, u):
    # Prediction step
    x = F @ x + G @ u  # State prediction
    P = F @ P @ F.T + Q  # Covariance prediction

    # Update step
    y = z - H @ x  # Measurement residual
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ y  # Updated state estimate
    P = (np.eye(2) - K @ H) @ P  # Updated covariance estimate

    return x, P

# Draw x and y axis
def draw_axis():
    pygame.draw.line(screen, (0, 0, 0), (0, height // 2), (width, height // 2), 2)  # x-axis
    pygame.draw.line(screen, (0, 0, 0), (width // 2, 0), (width // 2, height), 2)  # y-axis
    pygame.draw.circle(screen, (0, 0, 0), (width // 2, height // 2), 2) # origin 
    

# Main simulation loop
running = True
while running:
    screen.fill((255, 255, 255))  # Clear the screen
    

    # Draw x and y axis
    draw_axis()

    # Simulate the robot's movement
    u = np.array([[0.1], [0.1]])  # Control signals applied to the wheels
    x, P = kalman_filter(x, P, x + np.array([[0.5], [0]]), u)  # State estimation

    # Store ground truth and measured positions
    ground_truth_positions.append((int(x[0, 0] * 100) + width // 2, height // 2 - int(x[1, 0] * 100)))
    measured_positions.append((int(x[0, 0] * 100) + width // 2 + np.random.normal(0, 0.05),
                               height // 2 - int(x[1, 0] * 100) + np.random.normal(0, 0.075))) 

    # Draw the robot's position
    #pygame.draw.circle(screen, (255, 0, 0), (int(x[0, 0] * 100) + width // 2, height // 2 - int(x[1, 0] * 100)), 5)
    screen.blit(bot_img, (int(x[0, 0] * 100) + width // 2, int(x[1, 0] * 100) + height // 2))

	# Draw the ground truth trajectory
    if len(ground_truth_positions) >= 2:
        pygame.draw.lines(screen, (0, 255, 0), False, ground_truth_positions, 3)

    # Draw the measured trajectory
    if len(ground_truth_positions) >= 2:
        pygame.draw.lines(screen, (0, 0, 255), False, measured_positions, 3)
        
    # Draw legend
    legend_font = pygame.font.SysFont(None, 20)
    ground_truth_legend = legend_font.render('Ground Truth', True, (0, 255, 0))
    measured_legend = legend_font.render('Measured', True, (0, 0, 255))
    screen.blit(ground_truth_legend, (10, 10))
    screen.blit(measured_legend, (10, 30))
    
    pygame.display.flip()  # Update the display
    clock.tick(1)  # Limit the frame rate to 10 FPS
    
    if frame_count < 10:
        frame_img_filename = f'frame_{frame_count:03d}.png'
        pygame.image.save(screen, frame_img_filename)
        frame_images.append(frame_img_filename)
        frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
# Create GIF animation
gif_filename = 'animation.gif'
images = [imageio.imread(image) for image in frame_images]
imageio.mimsave(gif_filename, images, duration = 1000)

# Delete the PNG frames
for image in frame_images:
    os.remove(image)

# Quit Pygame
pygame.quit()
