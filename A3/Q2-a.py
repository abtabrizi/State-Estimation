"""
Created on Wed Jun  7 15:48:49 2023

@author: Arash Tabrizi

A3 Question 2(a) - nonLinear motion model + linear measurement model
"""
import pygame
import numpy as np
import matplotlib.pyplot as plt 
import random
import math 
import imageio.v2 as imageio
import os
from pygame.locals import *
from math import *

# Initialize PyGame
pygame.init()

# Set up the display & colors
width, height = 800, 700
gameDisplay = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# General initializations
crashed = False
position=np.array([[0],[0],[0]]) # initial state vector
coordinate_measure=np.array([[0],[0],[0]])
coordinate_gt=np.array([[0],[0],[0]])
coordinate_particle=np.array([[0],[0],[0]])
p_0=np.array([[0,0,0],[0,0,0],[0,0,0]])
flag_1=0 
t=1 # main loop counter
frame_images = []  # List to store frame image filenames
frame_count = 0 # counter for creating animation gif

# Load robot display image
bot_img = pygame.image.load('turtle.png')
my_robot_w = 30
my_robot_h = 30
bot_img = pygame.transform.scale(bot_img, (my_robot_w, my_robot_h))
bot_img = pygame.transform.rotate(bot_img, -90)

# Initializations for variables used in plotting
landmark=np.array([[10],[10]]) 
points=[]
points.append([400,400])
points_gt=[]
points_gt.append([400,400])
points_measure=[]
points_measure.append([400,400])
coordinate_measure_x=[]
coordinate_measure_y=[]
coordinate_pred_x=[]
coordinate_pred_y=[]
cov_width=[]
cov_hight=[]

# Function to draw r
def my_robot(x, y):
    gameDisplay.blit(bot_img, (x, y))

# Calculate the joint PDF for a two-dimensional Gaussian distanceribution
def joint_distance(x,y):

    # Fomulation to get Euclidean distance
    distance = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - coordinate_measure[0][0])**2/(2*0.005*0.005))+((y - coordinate_measure[1][0])**2/(2*0.075*0.075))))
    # Add a small value to avoid division by zero
    distance = distance + 1e-9 
    
    return distance

# Particle generation 
def create_particles():
    global position
    
    N_particles = 100 # change to samples (M)
    particles = []
    
    # Generate particles 
    particles_x = np.random.normal(position[0][0],0.0005,N_particles)
    particles_y = np.random.normal(position[1][0],0.0075,N_particles)
    particles_theta = np.random.normal(position[1][0],0,N_particles)
    
    # Create particle arrays and append to particles list
    for i in range(0, N_particles):
        particles.append(np.array([[particles_x[i]],[particles_y[i]],[particles_theta[i]]]))  
    
    return particles

# Calculate the weighted mean of particles
def get_weight(particles):

        distance=np.zeros((1,len(particles)))

        # Calculate the measurement probabilities for each particle
        for i in range(0,len(particles)):
            distance[0,i]=joint_distance(particles[i][0][0],particles[i][1][0])

        # Normalize the weights
        distance=distance/np.sum(distance)
        
        new_x = 0
        new_y = 0

        # Calculate the weighted mean
        for i in range(0, len(particles)):	
            new_x = new_x + distance[0,i] * particles[i][0][0]
            new_y = new_y + distance[0,i] * particles[i][1][0]

        coord=np.array([[new_x],[new_y],[0]])
 
        return coord, distance

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
                  
        resamp_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]],[particles[index][2][0]]]))
        
    return resamp_particles

# State transition definition (linear motion model)
def state_trans_model(position, particles):
    global coordinate_gt
    
    x=[]
    y=[]
    
    # State transition matrix
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])
    # Radius of wheels
    r=0.1
    # Time step
    T=1/8
    # Distance between wheels
    L=0.3
    # Control params 
    u_r=u_l=1
    
    if (math.dist([position[0,0],position[1,0]],landmark)<10):
        u_r=1
        u_l=0
    if (math.dist([position[0,0],position[1,0]],landmark)>11):
        u_r=0
        u_l=1
        
    # Control matrix
    B=np.array([[r*T*math.cos(position[2,0]),0],[r*T*math.sin(position[2,0]),0],[0,T*r/L]])
    B_gt=np.array([[r*T*math.cos(coordinate_gt[2,0]),0],[r*T*math.sin(coordinate_gt[2,0]),0],[0,T*r/L]])
    
    # Control vector (input)
    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    # Process noise vector
    w_k=np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
    
    # State transition equation 
    new_posi = np.matmul(A,position) + np.matmul(B,u) + w_k*T
    # State transition model ground truth (no noise)
    coordinate_gt = np.matmul(A,coordinate_gt) + np.matmul(B_gt,u)
    
    for i in range(0,len(particles)):

            if (math.dist([particles[i][0][0],particles[i][1][0]],landmark)<10):
                u_r=1
                u_l=0
            if (math.dist([particles[i][0][0],particles[i][1][0]],landmark)>11):
                u_r=0
                u_l=1
            
            # Control vector (input)
            u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
            
            temp = np.matmul(A,particles[i]) + np.matmul(B,u) + T*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
            particles[i] = temp
            
            x.append(temp[0][0])
            y.append(temp[1][0])
            
    return particles, new_posi, x, y
    
# Measurement model definition
def measurement_model():
    global coordinate_measure
    
    # Observation matrix
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    # Measurement noise vector
    n_k=np.asarray([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)],[0]])
    # Measurement equation | Eq (3) in solution PDF
    z=np.matmul(C,coordinate_gt) + n_k
    
    coordinate_measure=z

# Calculate mean and standard deviation of the coordinates
def get_stats(x,y):

    x=np.array(x)
    y=np.array(y)
    
    return np.mean(x),np.std(x),np.mean(y),np.std(y)

# Update function for cov matrix  
def get_cov():
    global p_0

    # State transition matrix
    A=np.array([[1,0,0],[0,1,0],[0,0,1]])
    temp=np.matmul(A,p_0)
    # Time step
    T=1/8
    # Process noise cov matrix
    Q=T*np.array([[0.01,0,0],[0,0.1,0],[0,0,0]])
    
    temp1=np.matmul(temp,A.transpose())
    new_cov=temp1+ Q
    
    p_0=new_cov
    
# Computing corrected observation matrix
def get_correction():
    global p_0

    # Observation matrix
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    # Measurement noise cov matrix | linear model
    R=np.array([[0.05,0,0],[0,0.075,0],[0,0,0]])
    
    temp1=np.matmul(p_0,C.transpose())
    temp2=np.matmul(np.matmul(C,p_0),C.transpose()) + R
    
    k=temp1/temp2
    k[np.isnan(k)] = 0

    return C,k

# Computing belief (x_t) at time t
def get_belief(C,K):
    global p_0 # Cov of elief  
    global position # Mean of belief   
    
    # Observation matrix
    C=np.array([[1,0,0],[0,2,0],[0,0,1]])
    
    # Compute mean of belief at time t
    position = position + np.matmul(K,(coordinate_measure - np.matmul(C,position)))
    
    # Compute cov of belief at time t
    p_0=np.matmul((np.identity(3)-np.matmul(K,C)),p_0)

################################ Main simulation loop ################################
while not crashed:
    
    # Quit simulation if simulation window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
    
    # Debugging line
    #print(t)
    
    if flag_1==0:
        particles=create_particles()
        flag_1=1
    
    particles,position,x,y=state_trans_model(position,particles)
    mean_x,std_x,mean_y,std_y=get_stats(x,y)
    get_cov()
    
    if(t%8==0):
        measurement_model()
        C_new,k_new=get_correction()
        get_belief(C_new,k_new)
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
    
    # Draw landmark point
    M = pygame.Surface((10, 10))
    M.fill((0, 0, 0))
    gameDisplay.blit(M, (400, 245))
    
    # Draw particles 
    for p in particles:        
        pygame.draw.rect(gameDisplay,RED,(p[0][0]*1000+400,(p[1][0]/2)*1000+400,2,2)) 
    
    # Draw cov ellipse
    radius = (coordinate_particle[0,0]*1000+400-(std_x*2000)/2, (coordinate_particle[1,0]/2)*1000+400-(2000*std_y)/2, 2000*std_x, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, BROWN, radius, 2)  
    
    # Draw robot using desired image
    my_robot(coordinate_particle[0,0]*1000+400, (coordinate_particle[1,0]/2)*1000+400)
    
    # Draw trajectories
    points.append([coordinate_particle[0,0]*1000+400,coordinate_particle[1,0]/2*1000+400])
    points_gt.append([coordinate_gt[0,0]*1000+400,coordinate_gt[1,0]*1000+400])
    #points_measure.append([coordinate_measure[0,0]*1000+400,(coordinate_measure[1,0]/2)*1000+400])
    pygame.draw.lines(gameDisplay,BLUE,False,points,5)
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5)
    #pygame.draw.lines(gameDisplay,RED,False,points_measure,5)

    # For plotting
    coordinate_measure_x.append(coordinate_measure[0,0])
    coordinate_measure_y.append(coordinate_measure[1,0])
    coordinate_pred_x.append(coordinate_particle[0,0])
    coordinate_pred_y.append(coordinate_particle[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    
    # Draw legend
    legend_font = pygame.font.SysFont(None, 30)
    ground_truth_legend = legend_font.render('Ground Truth', True, (0, 255, 0))
    pred_legend = legend_font.render('PF Estimation', True, (0, 0, 255))
    landmark_legend = legend_font.render('Landmark (M)', True, (0, 0, 0))
    #pos_meas_legend = legend_font.render('Measurement', True, (255, 0, 0))
    gameDisplay.blit(ground_truth_legend, (320, 520))
    gameDisplay.blit(pred_legend, (320, 540))
    gameDisplay.blit(landmark_legend, (320, 560))
    #gameDisplay.blit(pos_meas_legend, (750, 660))
    
    # Draw circle pointer for gt trajectory
    pygame.draw.circle(gameDisplay,GREEN,(coordinate_gt[0,0]*1000+400,(coordinate_gt[1,0])*1000+400),7,10)
    
    # Update the display
    pygame.display.update()
    # Limit the frame rate to 4 FPS
    clock.tick(8)
    
    ############################### End of Visualization Settings ###############################
    
    
    # Gif animation setup 
    if frame_count < 290: #190 check t at variable explorer
        frame_img_filename = f'frame_{frame_count:03d}.png'
        pygame.image.save(gameDisplay, frame_img_filename)
        frame_images.append(frame_img_filename)
        frame_count += 1
    
    # Update counter    
    t+=1 
    
# Plots 
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

plt.plot(cov_hight,label='Covariance of x values')
plt.plot(cov_width,label='Covariance of y values')
plt.xlabel("Iteation")
plt.ylabel("Value")
plt.legend()
plt.show()


# Create GIF animation
gif_filename = 'animation-2a.gif'
images = [imageio.imread(image) for image in frame_images]
imageio.mimsave(gif_filename, images, duration = 100)

# Delete the PNG frames
for image in frame_images:
    os.remove(image)

# Quit Pygame
pygame.quit()
