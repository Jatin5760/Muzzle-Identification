# analysis.py

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # IMPORTANT: Use a non-interactive backend for servers
import matplotlib.pyplot as plt
from skimage.filters import hessian
from skimage.feature import local_binary_pattern
import io
import base64

# --- This is our new helper function to convert plots to images ---
def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string with a dark theme."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0e1a')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# --- All of your functions are below, but MODIFIED to return images ---

def calculate_ridge_density(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = binary.shape[0] * binary.shape[1]
    ridge_length = 0
    min_contour_length = 20
    vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length > min_contour_length:
            ridge_length += length
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(vis_image, [contour], -1, (0, 0, 255), 1)
    ridge_density = ridge_length / total_area

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(blurred, cmap='viridis')
    axes[0].set_title('Blurred Image')
    axes[0].axis('off')
    axes[1].imshow(edges, cmap='viridis')
    axes[1].set_title('Edge Detection')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Ridge Contours (Density: {ridge_density:.6f})')
    axes[2].axis('off')
    fig.tight_layout()
    
    return ridge_density, fig_to_base64(fig)

def calculate_bead_arrangement_distance(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 40, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bead_distances = []
    vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            bead_distances.append((cX, cY))
            cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
            cv2.circle(vis_image, (cX, cY), 5, (0, 0, 255), -1)

    average_distance = 0.0
    if len(bead_distances) > 1:
        total_distance = 0
        for i in range(len(bead_distances) - 1):
            x1, y1 = bead_distances[i]
            x2, y2 = bead_distances[i + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        average_distance = total_distance / (len(bead_distances) - 1)

    mapped_value = (average_distance / 100.0) * 9.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(gray, cmap='viridis')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')
    axes[1].imshow(edges, cmap='viridis')
    axes[1].set_title('Edge Detection')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Bead Centroids\nAvg Distance: {average_distance:.2f}')
    axes[2].axis('off')
    fig.tight_layout()
    
    return int(mapped_value), fig_to_base64(fig)

def calculate_average_inter_ridge_distance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, threshed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(threshed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    vis_image = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
    average_distance = 0.0

    if lines is not None:
        inter_ridge_distances = []
        for line1 in lines:
            x1, y1, x2, y2 = line1[0]
            for line2 in lines:
                x3, y3, x4, y4 = line2[0]
                distance = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                inter_ridge_distances.append(distance)
        average_distance = np.mean(inter_ridge_distances)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(image, cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(threshed, cmap='viridis')
    axes[1].set_title('Thresholded Image')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Detected Lines\nAvg Inter-Ridge Distance: {average_distance:.2f}')
    axes[2].axis('off')
    fig.tight_layout()
    
    return average_distance, fig_to_base64(fig)

def average_bead_size(image_bgr):
    gray_muzzle = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_muzzle = cv2.threshold(gray_muzzle, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_muzzle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100
    beads = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
    
    average_size = 0
    if len(beads) > 0:
        bead_sizes = [cv2.contourArea(bead) for bead in beads]
        average_size = np.mean(bead_sizes)

    vis_image = image_bgr.copy()
    cv2.drawContours(vis_image, beads, -1, (0, 255, 0), 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(binary_muzzle, cmap='viridis')
    axes[1].set_title('Binary Image')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Detected Beads: {len(beads)}')
    axes[2].axis('off')
    fig.tight_layout()

    return average_size, fig_to_base64(fig)

def calculate_ridge_orientation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi = image[:, :image.shape[1] // 2]
    lbp_image = local_binary_pattern(roi, 8, 1, method='uniform')
    gradient_x = cv2.Sobel(lbp_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(lbp_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_xx = gradient_x * gradient_x
    gradient_yy = gradient_y * gradient_y
    gradient_xy = gradient_x * gradient_y
    kernel = cv2.getGaussianKernel(16, -1)
    kernel = kernel.dot(kernel.T)
    weighted_xx = cv2.filter2D(gradient_xx, -1, kernel)
    weighted_yy = cv2.filter2D(gradient_yy, -1, kernel)
    weighted_xy = cv2.filter2D(gradient_xy, -1, kernel)
    ridge_orientation = 0.5 * np.arctan2(2 * weighted_xy, weighted_xx - weighted_yy)
    ridge_orientation_degrees = (np.degrees(ridge_orientation) + 180) % 180
    num_orientations = 16
    bin_size = 180 / num_orientations
    orientation_histogram, bin_edges = np.histogram(ridge_orientation_degrees, bins=num_orientations, range=(0, 180))
    consensus_orientation = np.argmax(orientation_histogram) * bin_size

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(image, cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(roi, cmap='viridis')
    axes[1].set_title('ROI (Left Side)')
    axes[1].axis('off')
    axes[2].imshow(lbp_image, cmap='viridis')
    axes[2].set_title('LBP Image')
    axes[2].axis('off')
    bin_centers = bin_edges[:-1] + bin_size/2
    axes[3].bar(bin_centers, orientation_histogram, width=bin_size*0.8, alpha=0.7, color='#10b981')
    axes[3].axvline(x=consensus_orientation, color='#3b82f6', linestyle='--', label=f'Consensus: {consensus_orientation:.1f}°')
    axes[3].set_title('Ridge Orientation Histogram')
    axes[3].legend()
    fig.tight_layout()
    
    return float(consensus_orientation), fig_to_base64(fig)

def extract_bead_density(image):
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    num_labels, labeled_image = cv2.connectedComponents(binary_image)
    total_bead_area = np.sum(binary_image == 255)
    bead_density = num_labels / total_bead_area if total_bead_area > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0a0e1a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].imshow(image, cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(binary_image, cmap='viridis')
    axes[1].set_title('Adaptive Threshold')
    axes[1].axis('off')
    axes[2].imshow(labeled_image, cmap='viridis')
    axes[2].set_title(f'Connected Components\nLabels: {num_labels}, Density: {bead_density:.8f}')
    axes[2].axis('off')
    fig.tight_layout()
    
    return bead_density, fig_to_base64(fig)


# --- This is the main function that the web server will call ---
def analyze_single_image(image_path):
    """
    Analyzes an image and returns a dictionary of features and a dictionary of visualizations.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None, None
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    hessian_gray = hessian(image_gray)
    hessian_gray_uint8 = (hessian_gray * 255).astype(np.uint8)

    features = {}
    visualizations = {}
    
    # 1. Bead Arrangement Distance
    val, img_str = calculate_bead_arrangement_distance(image_path)
    features['Bead Arrangement Distance'] = val
    visualizations['Bead Arrangement'] = img_str
    
    # 2. Bead Density
    val, img_str = extract_bead_density(hessian_gray_uint8)
    features['Bead Density'] = round(val * 10e5)
    visualizations['Bead Density'] = img_str
    
    # 3. Ridge Orientation
    val, img_str = calculate_ridge_orientation(image_path)
    features['Ridge Orientation'] = val
    visualizations['Ridge Orientation'] = img_str

    # 4. Ridge Density
    val, img_str = calculate_ridge_density(image_gray)
    features['Ridge Density'] = round(val * 10e4)
    visualizations['Ridge Density'] = img_str

    # 5. Average Bead Size
    val, img_str = average_bead_size(image_bgr)
    features['Average Bead Size'] = round(val, 2)
    visualizations['Average Bead Size'] = img_str

    # 6. Average Inter-Ridge Distance
    val, img_str = calculate_average_inter_ridge_distance(image_path)
    features['Average Inter-Ridge Distance'] = round(val, 2)
    visualizations['Inter-Ridge Distance'] = img_str
    
    return features, visualizations