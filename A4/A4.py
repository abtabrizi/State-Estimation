# Required Python modules 
import numpy as np
import random
import matplotlib.pyplot as plt

# Estimating homography using RANSAC (algorithm explained in detail in A4 solution PDF)
def compute_homography(src_pts, dst_pts):
    num_iterations = 1000
    num_inliers_threshold = 10
    reprojection_error_threshold = 0.005

    best_num_inliers = 0
    best_homography = None
    best_inliers = None

    for i in range(num_iterations):
        # Randomly select four point correspondences
        indices = random.sample(range(len(src_pts)), 4)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # Compute homography using the four-point correspondences
        A = []
        for j in range(4):
            x, y = src_sample[j]
            u, v = dst_sample[j]
            A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

        # Using the NumPy function np.linalg.svd() to perform the SVD on the matrix A
        A = np.asarray(A)
        # The first two _ variables are used to discard the intermediate matrices U and Σ
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        # Calculate re-projection error
        # Creating homogeneous coordinates for the source points
        src_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        # Projecting the source points to the destination space
        dst_projected = src_homogeneous.dot(H.T)
        # Calculating the Euclidean distance between the projected points and the actual destination points
        dst_projected /= dst_projected[:, 2:]
        reproj_errors = np.linalg.norm(dst_pts - dst_projected[:, :2], axis=1)

        # Count inliers
        # Counting the number of inliers based on the re-projection errors
        inliers = np.where(reproj_errors < reprojection_error_threshold)[0]
        num_inliers = len(inliers)

        # Update the best model if necessary
        if num_inliers > best_num_inliers and num_inliers >= num_inliers_threshold:
            best_num_inliers = num_inliers
            best_homography = H
            best_inliers = inliers

    return best_homography, best_inliers

# Source and destination coordinates
src_pts = np.array([[1.90659, 2.51737], [2.20896, 1.1542], [2.37878, 2.15422], [1.98784, 1.44557], [2.83467, 3.41243], [9.12775, 8.60163], [4.31247, 5.57856], [6.50957, 5.65667], [3.20486, 2.67803], [6.60663, 3.80709], [8.40191, 3.41115], [2.41345, 5.71343], [1.04413, 5.29942], [3.68784, 3.54342], [1.41243, 2.6001]])
dst_pts = np.array([[5.0513, 1.14083], [1.61414, 0.92223], [1.95854, 1.05193], [1.62637, 0.93347], [2.4199, 1.22036], [5.58934, 3.60356], [3.18642, 1.48918], [3.42369, 1.54875], [3.65167, 3.73654], [3.09629, 1.41874], [5.55153, 1.73183], [2.94418, 1.43583], [6.8175, 0.01906], [2.62637, 1.28191], [1.78841, 1.0149]])

# Compute homography using RANSAC
homography, inliers = compute_homography(src_pts, dst_pts)

# Apply the homography to the source points (the following 3 lines are used to generate figure 3 of A4 solution PDF)
transformed_pts = np.dot(homography, np.column_stack((src_pts, np.ones((len(src_pts), 1)))).T)
transformed_pts = transformed_pts[:2, :] / transformed_pts[2, :]
transformed_pts = transformed_pts.T
# This line plots the transformed source points (as seen in figure 3 of A4 solution PDF)
#plt.plot(transformed_pts[:, 0], transformed_pts[:, 1], 'bo', markersize=8, fillstyle='none', label='Transformed Source')
# Debuggin line that tells us how many source points were transformed 
#print("Number of Transformed Points:", len(transformed_pts))

# Print the normalized homography transformation
print("Normalized Homography Transformation:")
print(homography / homography[2, 2])

# Plotting
plt.figure(figsize=(12, 10))
plt.plot(src_pts[:, 0], src_pts[:, 1], 'yo', label='Source')
plt.plot(dst_pts[:, 0], dst_pts[:, 1], 'ro', label='Destination')
plt.plot(src_pts[inliers, 0], src_pts[inliers, 1], 'o', color='black', fillstyle='none', markersize=8, label='Inliers')
outliers = np.delete(np.arange(len(src_pts)), inliers)
plt.plot(src_pts[outliers, 0], src_pts[outliers, 1], 'x', color='black', markersize=7, label='Outliers')

# Connect inliers with dashed green lines
for i in inliers:
    plt.plot([src_pts[i, 0], dst_pts[i, 0]], [src_pts[i, 1], dst_pts[i, 1]], 'g--')
    

# Other plot settings 
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
# Turn on the grid
plt.grid(True)
# Axis limits for better visualization 
plt.xlim(1, 9.5)
plt.ylim(0, 8.7)
plt.title('RANSAC Homography Estimation')
plt.show()
