import pygame
import math
import numpy as np
from numpy import dot, array, eye, zeros, diag
from numpy.linalg import inv, norm
import imageio.v2 as imageio
import os

# User settings
Noise = True 
Linear = False # set to 'True' when simulating part (a)
Ellipse = True 
if Linear == True:
    part = "a"
else:
    part = "b"

# Initialize PyGame
pygame.init()

# Set up the display & colors
res = 800
gameDisplay = pygame.display.set_mode((res, res))
sc_f = int(res/20)
cov_size = 20  # this should be changed to 300 if 'Linear' is set to 'True'

# Draw landmark at (10, 10)
Landmark = pygame.Surface((10, 10))
Landmark.fill((0, 0, 0))
L = array([[10, 10]]).T

# More display settings
font = pygame.font.Font(None, 25)

# Legend location and settings  
textX = 10
textY = 10
gameDisplay.fill((255, 255, 255))
gameDisplay.blit(Landmark, (int(res/2), int(res/2)))
Q_text = font.render("Part (" + str(part) + ") Simulation", True, (0, 0, 0))
gameDisplay.blit(Q_text, (textX, textY+20))


# Draw robot
robot = pygame.Surface((4, 4))
robot.fill((255, 0, 0))

# Covariance ellipse settings
image = pygame.Surface([res,res], pygame.SRCALPHA, 32)
image = image.convert_alpha()

# We start robot at (6, 10), this is arbitrary 
x0 = 6.0 
y0 = 10.0

# Initializations
theta = math.pi/2.0
X = array([[x0, y0]]).T
P = zeros(2)

# Motion model setup
r = 0.1
rL = 0.2
T = float(1/8)
A = eye(2) # A is state transition matrix
ur = 4.75
ul = 5.25
u_w = (ur+ul)/2.0
u_phi = (ur-ul)*(r/rL)
U = array([[r*u_w*math.cos(theta),r*u_w*math.sin(theta)]]).T

# Process Noise (motion model)
ww = 0.1
wphi = 0.01
Q = diag([ww+wphi, ww+wphi])
delf_delw = array([T, T])
delf_delphi = (T**2)*r*u_w* array([-math.sin(theta), math.cos(theta)])
w_k = array([ww*delf_delw[0]+wphi*delf_delphi[0], ww*delf_delw[1]+wphi*delf_delphi[1]])
Q_p = diag([w_k[0], w_k[1]])

# Measurement setup 
C = diag([1, 2])
rx, ry = 0.05, 0.075
R = diag([rx, ry])
phi = theta + math.pi/2.0
dist_norm = 4.0

# Other initializations
i = 0
running = True
frames = [] # to store image frames

# True line
true_line = [(x0*sc_f, y0*sc_f), (x0, y0)]

# KF predictor definition
def kf_predict(X, P, A, Q_p, U):
    noise_x = np.random.normal(0, ww)
    noise_y = np.random.normal(0, ww)
    noise = array([[noise_x, noise_y]]).T
    X_dot = U + noise
    X = dot(A, X) + (T*X_dot)
    P = dot(A, dot(P, A.T)) + Q_p
    return X, P

# KF corrector definition for part (a)
def kf_correct(X, P, Y, C, R):
    IM = dot(C, X)
    IS = R + dot(C, dot(P, C.T))
    K = dot(P, dot(C.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(C, P))
    return (X, P, K, IM, IS)

# KF corrector definition for part (b)
def kf_correct_part_b(X, P, Y, G, R_p, IM):
    IS = R_p + dot(G, dot(P, G.T))
    det = IS[0,0]*IS[1,1] - IS[0,1]*IS[1,0]
    #try to clip outliers
    inverse = inv(IS)
    if abs(det) < 0.1:
        inverse = (abs(det)/2)*inv(IS)
    elif abs(det) < 0.2:
        inverse = abs(det)*inv(IS)
    elif abs(det) < 0.3:
        inverse = abs(det)*1.5*inv(IS)
    elif abs(det) < 0.4:
        inverse = abs(det)*2*inv(IS)
    else:
        inverse = inv(IS)
    K = dot(P, dot(G.T, inverse))
    X = X + dot(K, (Y - IM))
    P = (P - dot(K, dot(G, P)))
    #P = (eye(2) - dot(K, C)) * P
    return (X, P, K, IM, IS)


def range_bearing(d_norm, phi):
    return array([10+(d_norm*math.cos(phi)), 10+(d_norm*math.sin(phi))])

# No noise simulation 
if Noise == False:
    ww, wphi, rx, ry = 0, 0, 0, 0

# Main simulation loop
while running:
    i += 1 
    
    # Capture the screen as image frames
    frame = pygame.surfarray.array3d(gameDisplay)
    frames.append(frame)
    
    # End simulation if simulation window closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.time.wait(125)
    
    theta_noise = np.random.normal(0, wphi)
    theta_dot = (r/rL)*u_phi + theta_noise
    theta = (T*theta_dot) + theta
    theta_no_noise = (T*(r/rL)*u_phi) + theta
    U = array([[r*u_w*math.cos(theta),r*u_w*math.sin(theta)]]).T

    delf_delphi = (T**2)* array([-r*u_w*math.sin(theta_no_noise), r*u_w*math.cos(theta_no_noise)])
    w_k = array([ww*delf_delw[0]+wphi*delf_delphi[0], ww*delf_delw[1]+wphi*delf_delphi[1]])
    Q_p = diag([w_k[0], w_k[1]])

    X, P = kf_predict(X, P, A, Q_p, U)

    if i%8 == 0:
        if Linear == True:
            n_x = np.random.normal(0, rx)
            n_y = np.random.normal(0, ry)
            n = diag([n_x, n_y])
            Z = dot(C, X) + n
            X, P, K, IM, IS = kf_correct(X, P, Z, C, R)
            true_line[1] = (int(X[0][0]*sc_f), int(X[1][0]*sc_f))
            pygame.draw.lines(gameDisplay, (0, 0, 255), False, true_line, 3)
            true_line[0] = true_line[1]
        else:
            dist = X - L
            dist = dist[:, 0]
            n_w = np.random.normal(0, ww)
            n_phi = np.random.normal(0, wphi)
            dist_norm = norm(dist, ord = 2)
            dist_norm_n = dist_norm + n_w
            phi = theta + math.pi/2.0
            phi_n = phi + n_phi
            

            # Update R
            delg_delw = array([math.cos(phi), math.sin(phi)])
            delg_delphi = dist_norm* array([-math.sin(phi), math.cos(phi)])
            

            n_k = array([ww*delg_delw[0]+wphi*delg_delphi[0], ww*delg_delw[1]+wphi*delg_delphi[1]])
            R_p = diag([n_k[0], n_k[1]])

            Z = range_bearing(dist_norm_n, phi_n)
            IM = range_bearing(dist_norm, phi)
            G1 = array([(X[0, 0] - 10.0)*math.cos(phi), (X[1,0] - 10.0)*math.cos(phi)])
            G2 = array([(X[0, 0] - 10.0)*math.sin(phi), (X[1,0] - 10.0)*math.sin(phi)])
            G = (1.0/dist_norm)*array([G1, G2])
            X, P, K, IM, IS = kf_correct_part_b(X, P, Z, G, R_p, IM)
            true_line[1] = (int(X[0, 0]*sc_f), int(X[1, 0]*sc_f))
            pygame.draw.lines(gameDisplay, (0, 0, 255), False, true_line, 3)
            true_line[0] = true_line[1]

    Xx = (X[0][0]*sc_f)
    Xy = (X[1][0]*sc_f)

    Px = (P[0][0])
    Py = (P[1][1])

    dist = X - L
    dist = dist[:,0]
    dist_norm = norm(dist, ord = 2)

    
    if theta < - 0:
        theta = math.pi*2.0

    gameDisplay.blit(robot, (int(Xx), int(Xy)))

    if Ellipse == True:
        try:
            #gameDisplay.fill((255, 255, 255)) # un-comment if we want to visual real-time cov ellipse updates only
            pygame.draw.ellipse(gameDisplay, (0, 255, 0), (int(Xx-(Px*cov_size/2)), int(Xy-(Py*cov_size/2)), int(Px*cov_size), int(Py*cov_size)), 1)

        except:
            print('We are here!')

    n_phi = np.random.normal(0, wphi)
    Bearing = (theta + math.pi/2.0 + n_phi)*180.0/math.pi
    if (theta*180.0/math.pi) > 270:
        Bearing = Bearing - 360
    Bearing_print = '{:03.2f}'.format(Bearing)
    B_text = font.render("Bearing: " + Bearing_print, True, (0, 0, 0))
    gameDisplay.fill((255, 255, 255), (0, res//12, res, res//6))
    gameDisplay.blit(Landmark, (int(res/2), int(res/2)))
    gameDisplay.blit(B_text, (textX, textY+60))


    pygame.display.update()

# Create animation GIF using imageio package 
frames = [np.flip(frame, axis=1) for frame in frames]
frames = [np.rot90(frame, k=1) for frame in frames]
imageio.mimsave('animation-2b.gif', frames, duration=125)

# Delete the PNG frames
for file_name in os.listdir('.'):
    if file_name.endswith('.png'):
        os.remove(file_name)

# Quit Pygame
pygame.quit()