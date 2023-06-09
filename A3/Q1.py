"""
Created on Tue Jun  6 23:50:51 2023

@author: Arash Tabrizi

A3 Question 1 - Linear motion & observation models with PF
"""
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt 
import random
import imageio.v2 as imageio
import os

# Initialize PyGame
pygame.init()

# Set up the display & colors
width, height = 1000, 750
gameDisplay = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
white = (255,255,255)

# General initializations
crashed = False
position=np.array([[0],[0]]) # initial state vector
coordinate_measure=position
coordinate_gt=position
coordinate_particle=np.array([[0],[0]])

flag_1=0 
t=1 # main loop counter
frame_images = []  # List to store frame image filenames
frame_count = 0 # counter for creating animation gif

# Load robot display image
bot_img = pygame.image.load('turtle.png')
my_robot_w = 40
my_robot_h = 40
bot_img = pygame.transform.scale(bot_img, (my_robot_w, my_robot_h))
bot_img = pygame.transform.rotate(bot_img, -90)

# Initializations for variables used in plotting
points=[]
points.append([0,0])
points_gt=[]
points_gt.append([0,0])
points_measure=[]
points_measure.append([0,0])
coordinate_measure_x=[]
coordinate_measure_y=[]
coordinate_pred_x=[]
coordinate_pred_y=[]

# Function to draw r
def my_robot(x, y):
    gameDisplay.blit(bot_img, (x, y))

# Calculate the joint PDF for a two-dimensional Gaussian distanceribution
def joint_distance(x,y):

    # Fomulation
    distance = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - coordinate_measure[0][0])**2/(2*0.005*0.05))+((y - coordinate_measure[1][0])**2/(2*0.075*0.075))))
    # Add a small value to avoid division by zero
    distance = distance + 1e-9 
    
    return distance

# Particle generation 
def create_particles():
    global position
    
    N_particles = 100
    particles = []
    
    # Generate particles by sampling from a Gaussian distanceribution
    particles_x = np.random.normal(position[0][0], 0.005, N_particles)
    particles_y = np.random.normal(position[1][0], 0.075, N_particles)
    
    # Create particle arrays and append to particles list
    for i in range(0, N_particles):
        particles.append(np.array([[particles_x[i]], [particles_y[i]]]))  
    
    return particles

# Madow's resampling algorithm 
def resampling(particles, distance):
    
    N=len(particles)
    resamp_particles = []
    index = int(random.random() * N)
    beta = 0.0
    
    for i in range(0,N):
        distance[0,i]=joint_distance(particles[i][0][0],particles[i][1][0])
        
    distance=distance/np.sum(distance)
    mw = distance.max()

    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > distance[0][index]:
            beta -= distance[0][index]
            index = (index + 1) % N
                  
        resamp_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]]]))
        
    return resamp_particles

# Calculate the weighted mean of particles
def get_weight(particles):

        distance=np.zeros((1,len(particles)))

        new_x = 0
        new_y = 0
        
        # Calculate the measurement probabilities for each particle
        for i in range(0,len(particles)):
            distance[0,i]=joint_distance(particles[i][0][0],particles[i][1][0])

        # Normalize the weights
        distance=distance/np.sum(distance)

        # Calculate the weighted mean
        for i in range(0, len(particles)):	
            new_x = new_x + distance[0,i] * particles[i][0][0]
            new_y = new_y + distance[0,i] * particles[i][1][0]

        coord=np.array([[new_x],[new_y]])
 
        return coord, distance

# State transition definition (linear motion model)
def state_trans_model(position, particles):
    global coordinate_gt
    x=[]
    y=[]
    
    # State transition matrix
    A=np.array([[1,0],[0,1]])
    # Radius of wheels
    r=0.1
    # Time step
    T=1/8
    # Control matrix
    B=np.array([[(r/2)*T,(r/2)*T],[(r/2)*T,(r/2)*T]])
    # Control vector 
    u=np.array([[1],[1]])
    # Process noise vector
    w_k=np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])
    
    # State transition equation 
    new_posi = np.matmul(A,position) + np.matmul(B,u) + w_k*T
    # State transition model ground truth (no noise)
    coordinate_gt = np.matmul(A,coordinate_gt)+np.matmul(B,u)
    
    for i in range(0,len(particles)):

            temp = np.matmul(A,particles[i]) + np.matmul(B,u) + T*np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])
            particles[i]=temp
            x.append(temp[0][0])
            y.append(temp[1][0])
            
    return particles, new_posi, x, y
    
# Measurement model definition
def measurement_model():
    global coordinate_measure
    
    # Observation matrix
    C=np.array([[1,0],[0,2]])
    # Measurement noise vector
    n_k=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)]])
    # Measurement equation | Eq (3) in solution PDF
    z=np.matmul(C,coordinate_gt) + n_k
    
    coordinate_measure=z   

# Calculate mean and standard deviation of the coordinates
def get_stats(x,y):

    x=np.array(x)
    y=np.array(y)
    
    return np.mean(x),np.std(x),np.mean(y),np.std(y)

################################ Main simulation loop ################################
while not crashed:
    
    # Quit simulation if simulation window closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
    
    # Stop simulation after certain distanceance reached
    if t >= 60: 
        break
    
    if flag_1==0:
        particles=create_particles()
        flag_1=1
    
    particles,position,x,y=state_trans_model(position,particles)
    mean_x,std_x,mean_y,std_y=get_stats(x,y)
    
    if(t%8==0):
        measurement_model()
        coordinate_particle,distance=get_weight(particles)
        particles=resampling(particles, distance)

    ##################################### Visualization Settings ##################################### 
    # Color settings
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    BROWN=(180,50,50)
    WHITE=(255,255,255)
    
    # Clear the screen
    gameDisplay.fill(WHITE)
    
    # Draw particles in color RED
    for p in particles:        
        pygame.draw.rect(gameDisplay,RED,(p[0][0]*1000+50,(p[1][0]/2)*1000+50,2,2))
    
    # Draw cov ellipse
    radius = (mean_x*1000+50-(std_x*2000)/2, mean_y/2*1000+50-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, BROWN, radius, 2)  
    
    # Draw robot using desired image
    my_robot(coordinate_particle[0,0]*1000+50, (coordinate_particle[1,0]/2)*1000+50)
    
    # Draw trajectories
    points.append([coordinate_particle[0,0]*1000+50,coordinate_particle[1,0]/2*1000+50])
    points_gt.append([coordinate_gt[0,0]*1000+50,coordinate_gt[1,0]*1000+50])
    points_measure.append([coordinate_measure[0,0]*1000+50,(coordinate_measure[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,BLUE,False,points,5)
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5)
    pygame.draw.lines(gameDisplay,RED,False,points_measure,5)

    coordinate_measure_x.append(coordinate_measure[0,0])
    coordinate_measure_y.append(coordinate_measure[1,0])
    coordinate_pred_x.append(coordinate_particle[0,0])
    coordinate_pred_y.append(coordinate_particle[1,0])
    
    # Draw legend
    legend_font = pygame.font.SysFont(None, 30)
    ground_truth_legend = legend_font.render('Ground Truth', True, (0, 255, 0))
    pred_legend = legend_font.render('Mean Particle Position', True, (0, 0, 255))
    pos_meas_legend = legend_font.render('Measurement', True, (255, 0, 0))
    gameDisplay.blit(ground_truth_legend, (750, 620))
    gameDisplay.blit(pred_legend, (750, 640))
    gameDisplay.blit(pos_meas_legend, (750, 660))
    
    # Update the display
    pygame.display.update()
    # Limit the frame rate to 4 FPS
    clock.tick(8)
    
    ############################### End of Visualization Settings ###############################
    
    
    # Gif animation setup 
    if frame_count < 60:
        frame_img_filename = f'frame_{frame_count:03d}.png'
        pygame.image.save(gameDisplay, frame_img_filename)
        frame_images.append(frame_img_filename)
        frame_count += 1
    
    # Update counter    
    t+=1 
    
# Plots 
plt.plot(coordinate_measure_x, coordinate_measure_y, label='Measurement')
plt.plot(coordinate_pred_x, coordinate_pred_y, label='Mean Particle Position')
plt.xlabel("Iteation")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.plot(coordinate_measure_x,label='x values')
plt.plot(coordinate_measure_y,label='y values')
plt.xlabel("Iteation")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.plot(coordinate_pred_x,label='x values')
plt.plot(coordinate_pred_y,label='y values')
plt.xlabel("Iteation")
plt.ylabel("Value")
plt.legend()
plt.show()


# Create GIF animation
gif_filename = 'animation-1.gif'
images = [imageio.imread(image) for image in frame_images]
imageio.mimsave(gif_filename, images, duration = 120)

# Delete the PNG frames
for image in frame_images:
    os.remove(image)

# Quit Pygame
pygame.quit()


