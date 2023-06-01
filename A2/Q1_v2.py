import pygame
import numpy as np
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
import os

# Initialize PyGame
pygame.init()

# Set up the display & colors
width, height = 1000, 600
gameDisplay = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
white = (255,255,255)

# Initializations
crashed = False
points=[]
points.append([0,0])
points_measure=[]
points_gt=[]
points_gt.append([0,0])
points_measure.append([0,0])
measured_positions_x=[]
measured_positions_y=[]
predicted_positions_x=[]
predicted_positions_y=[]
cov_width=[]
cov_hight=[]
position=np.array([[0],[0]]) # initial state vector
position_measure=position
p_0=np.array([[0,0],[0,0]]) # initial cov matrix
point_prev=[[0],[0]]
distance_trav = 0.0 # initial distance
t=1 # main loop counter
frame_images = []  # List to store frame image filenames
frame_count = 0 # counter for creating animation gif

# Robot display image
bot_img = pygame.image.load('turtle.png')
car_w = 40
car_h = 40
bot_img = pygame.transform.scale(bot_img, (car_w, car_h))
bot_img = pygame.transform.rotate(bot_img, -90)

# Function to assign image to robot
def car(x, y):
    gameDisplay.blit(bot_img, (x, y))

# State transition definition 
def state_trans_model(position):
    global new_posi_gt
    
    # State transition matrix
    A=np.array([[1,0],[0,1]])
    # Radius of wheels
    r=0.1
    # Time step
    T=1/8
    # Control matrix
    B=np.array([[(r/2)*T,0],[(r/2)*T,0]])
    # Control vector 
    u=np.array([[1],[1]])
    # Process noise vector
    w_k=np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])
    
    # State transition equation | Eq (10) in solution PDF
    new_posi = np.matmul(A,position) + np.matmul(B,u) + w_k*T
    # State transition model ground truth (no noise)
    # Line 2 of Kalman Filter algorithm in Prob Robotics textbook (Table 3.1)
    new_posi_gt = np.matmul(A,position)+np.matmul(B,u)
    
    return new_posi

# Computing state cov matrix
def get_new_cov():
    global p_0
    
    # State transition matrix
    A=np.array([[1,0],[0,1]])
    # Time step
    T=1/8
    # Computing A * Cov
    temp=np.matmul(A,p_0)
    # Process noise cov matrix
    Q=np.array([[0.1,0],[0,0.15]])
    # Computing new Cov 
    # Line 3 of Kalman Filter algorithm in Prob Robotics textbook (Table 3.1)
    new_cov=np.matmul(temp,A.transpose())+Q*T
    p_0=new_cov
    
# Measurement model definition
def measurement_model():
    global position_measure
    global position
    
    # Observation matrix
    C=np.array([[1,0],[0,2]])
    # Measurement noise vector
    n_k=np.array([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)]])
    # Measurement equation | Eq (3) in solution PDF
    z=np.matmul(C,new_posi_gt) + n_k
    
    position_measure=z

# Computing Kalman Filter gain  
def get_gain():
    global p_0

    # Observation matrix
    C=np.array([[1,0],[0,2]])
    # Measurement noise cov matrix 
    R=np.array([[0.05,0],[0,0.075]])
    # Computing filter gain
    # Line 4 of Kalman Filter algorithm in Prob Robotics textbook (Table 3.1)
    temp1=np.matmul(p_0,C.transpose())
    temp2=np.matmul(np.matmul(C,p_0),C.transpose()) + R
    k=temp1/temp2
    k[np.isnan(k)] = 0

    return C,k

# Computing belief (x_t) at time t
def get_belief(C,K):
    global p_0 # Cov of elief  
    global position # Mean of elief   
    # Observation matrix
    C=np.array([[1,0],[0,2]])
    # Compute mean of belief at time t
    # Line 5 of Kalman Filter algorithm in Prob Robotics textbook (Table 3.1)
    position = position + np.matmul(K,(position_measure-np.matmul(C,position)))
    # Compute cov of belief at time t
    # Line 6 of Kalman Filter algorithm in Prob Robotics textbook (Table 3.1)
    p_0=np.matmul((np.identity(2)-np.matmul(K,C)),p_0)
    #print("Position: ", position) # Debugging line

# Main simulation loop
while not crashed:
    # Quit simulation if simulation window closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    position=state_trans_model(position)
    get_new_cov()
    distance_trav += np.linalg.norm(np.array([position[0][0], position[1][0]]) - np.array([point_prev[0][0], point_prev[1][0]]))
    
    # Stop simulation after certain distance reached
    if distance_trav >= 10.0: 
        break

    if(t%8==0):
        measurement_model()
        C_new,K_new=get_gain()
        get_belief(C_new,K_new)

    # Clear the screen
    gameDisplay.fill(white)
    # Color settings
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    BROWN=(180,50,50)
    
    # Draw cov ellipse
    size = (position[0,0]*1000+50-(p_0[0,0]*2000)/2, position[1,0]*1000+50-(2000*p_0[1,1])/2, p_0[0,0]*2000, 2000*p_0[1,1])
    pygame.draw.ellipse(gameDisplay, BROWN, size,1)  
    
    pygame.draw.polygon(gameDisplay, BLUE,
                        [[position[0,0]*1000+50,position[1,0]*1000+50],[position[0,0]*1000+40,position[1,0]*1000+35] ,
                       [position[0,0]*1000+40,position[1,0]*1000+65]])
    
    # Draw robot using desired image
    car(position[0,0]*1000+50, position[1,0]*1000+50)
    #car(position[0,0]*1000+50, 10)
    
    points.append([position[0,0]*1000+50,position[1,0]*1000+50])
    points_gt.append([new_posi_gt[0,0]*1000+50,new_posi_gt[1,0]*1000+50])
    points_measure.append([position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,BLUE,False,points,5)
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5)
    pygame.draw.lines(gameDisplay,RED,False,points_measure,5)



    if(t%8==0):
        pygame.draw.rect(gameDisplay,RED,(position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50,10,10))
        pygame.draw.rect(gameDisplay,GREEN,(new_posi_gt[0,0]*1000+50,(new_posi_gt[1,0]/2)*1000+50,10,10))

    pygame.draw.rect(gameDisplay,RED,(position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50,10,10))
    pygame.draw.rect(gameDisplay,GREEN,(new_posi_gt[0,0]*1000+50,(new_posi_gt[1,0])*1000+50,10,10))   
    measured_positions_x.append(position_measure[0,0])
    measured_positions_y.append(position_measure[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    
    # Draw legend
    legend_font = pygame.font.SysFont(None, 30)
    ground_truth_legend = legend_font.render('Ground Truth', True, (0, 255, 0))
    pred_legend = legend_font.render('Prediction', True, (0, 0, 255))
    pos_meas_legend = legend_font.render('Measurement', True, (255, 0, 0))
    gameDisplay.blit(ground_truth_legend, (850, 520))
    gameDisplay.blit(pred_legend, (850, 540))
    gameDisplay.blit(pos_meas_legend, (850, 560))
    
    # Update the display
    pygame.display.update()
    # Limit the frame rate to 4 FPS
    clock.tick(4)
    
    if frame_count < 60:
        frame_img_filename = f'frame_{frame_count:03d}.png'
        pygame.image.save(gameDisplay, frame_img_filename)
        frame_images.append(frame_img_filename)
        frame_count += 1
    
    # Update counter    
    t+=1 
    
# Plot settings for trajectories
plt.plot(measured_positions_x,label='x values')
plt.plot(measured_positions_y,label='y values')
plt.xlabel("Iteation")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.plot(predicted_positions_x,label='x values')
plt.plot(predicted_positions_y,label='y values')
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
gif_filename = 'animation.gif'
images = [imageio.imread(image) for image in frame_images]
imageio.mimsave(gif_filename, images, duration = 300)

# Delete the PNG frames
for image in frame_images:
    os.remove(image)

# Quit Pygame
pygame.quit()

