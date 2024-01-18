import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
import numpy as np
from scipy import ndimage

# Define camera positions (x, y), FOV angles (degrees), and camera image path
LOGO_PATH = './logo/'
cameras = [
    {'position': (0, 0), 'orientation': -80,  'fov': 60, 'max_distance': 6, 'color': 'cornflowerblue',  'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_letage'},
    {'position': (0, 0), 'orientation': -80,  'fov': 60, 'max_distance': 6, 'color': 'cornflowerblue', 'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_airbnb_terreaux1'},
    {'position': (0, 0), 'orientation': -80,  'fov': 60, 'max_distance': 6, 'color': 'cornflowerblue', 'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_airbnb_terreaux2'},
    {'position': (0, 0), 'orientation': 185,  'fov': 40, 'max_distance': 8, 'color': 'cornflowerblue',  'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_bell_tower'},
    {'position': (0, 0), 'orientation': 235,  'fov': 50, 'max_distance': 6, 'color': 'cornflowerblue',  'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_city_hall_corner'},
    {'position': (0, 0), 'orientation': 100,  'fov': 60, 'max_distance': 6, 'color': 'cornflowerblue',  'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_airbnb_constantine1'},
    {'position': (0, 0), 'orientation': 100,  'fov': 60, 'max_distance': 6, 'color': 'cornflowerblue',  'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_airbnb_constantine2'},
    {'position': (0, 0), 'orientation': -10,  'fov': 90, 'max_distance': 8, 'color': 'cornflowerblue', 'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_saint_jean1'},
    {'position': (0, 0), 'orientation': -10,  'fov': 90, 'max_distance': 8, 'color': 'cornflowerblue', 'image_path': LOGO_PATH + 'camera_logo.png', 'name': 'cam_saint_jean2'}
]

def draw_camera_fov(ax, camera):
    pos = camera['position']
    fov = camera['fov']
    rotation = camera['orientation'] # Rotation angle in degrees
    max_distance = camera['max_distance'] # Maximum distance for FOV representation
    col = camera['color']

    num_sectors=200
    # Divide the FOV into sectors and assign a decreasing opacity to each
    for i in range(num_sectors):
        start_distance = max_distance * (i / num_sectors)
        end_distance = max_distance * ((i + 1) / num_sectors)
        alpha = 1/(1+np.exp(start_distance/3)) 
        # Create a wedge representing the FOV with a color gradient for each sector
        wedge = Wedge(pos, end_distance, -fov/2 + rotation, fov/2 + rotation, width=end_distance-start_distance, color=col, alpha=alpha)
        ax.add_patch(wedge)


def add_camera_logo(ax, camera):
    image_path = camera['image_path']
    pos = camera['position']
    rotation = camera['orientation'] # Rotation angle in degrees
    
    # Read the image
    img = plt.imread(image_path)
    rotated_img = ndimage.rotate(img, rotation)  # Rotate the image by 45 degrees
    img_norm = (rotated_img - rotated_img.min()) / (rotated_img.max() - rotated_img.min()) # Normalize the image to be between 0 and 1
    mask = img_norm[:,:,3] < 0.5 # Create a mask for the transparent pixels
    img_norm[mask] = 0
    # Create an annotation box with the camera logo
    imagebox = OffsetImage(img_norm, zoom=0.1)
    ab = AnnotationBbox(imagebox, pos, frameon=False)
    ax.add_artist(ab)

for cam in cameras:
    name = LOGO_PATH + cam['name'] + '.png'
    fig, ax = plt.subplots()
    add_camera_logo(ax,cam)
    draw_camera_fov(ax, cam)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(name, format='png',transparent=True)
    plt.close()

