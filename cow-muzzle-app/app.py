# app.py

import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import hessian
from skimage.feature import local_binary_pattern
import pandas as pd
import plotly.express as px

# --- (The first part of the file is the same) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_DIR = os.path.dirname(APP_DIR)
SAM2_DIR = os.path.join(PROJECT_DIR, 'segment-anything-2'); sys.path.append(SAM2_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
UPLOAD_FOLDER = os.path.join(APP_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DB_PATH = os.path.join(SAM2_DIR, 'database.json')
SAM_MODEL_PATH = os.path.join(SAM2_DIR, 'best_sam2_model.pth')
SAM_CFG_NAME = 'sam2_hiera_s.yaml'
app = Flask(__name__); app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("Loading AI models, this may take a moment..."); device = torch.device("cpu")
original_cwd = os.getcwd()
try:
    os.chdir(SAM2_DIR)
    sam2_model = build_sam2(SAM_CFG_NAME, checkpoint_path=None, device=device)
    checkpoint = torch.load(SAM_MODEL_PATH, map_location=device)
    sam2_model.load_state_dict(checkpoint['model_state_dict'])
    sam_predictor = SAM2ImagePredictor(sam2_model)
finally:
    os.chdir(original_cwd)
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1]); resnet_feature_extractor.eval()
print("All models loaded successfully.")
def predict_muzzle_mask(image_rgb):
    sam_predictor.set_image(image_rgb); height, width, _ = image_rgb.shape
    box_prompt = np.array([0, 0, width, height])
    masks, scores, _ = sam_predictor.predict(box=box_prompt, multimask_output=True)
    return masks[np.argmax(scores)]
def preprocess_image_for_resnet(image_array):
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    return preprocess(image_array).unsqueeze(0)
def extract_fingerprint(original_image, mask):
    binary_mask = (mask > 0).astype(np.uint8); muzzle_only_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
    image_tensor = preprocess_image_for_resnet(muzzle_only_image)
    with torch.no_grad(): fingerprint = resnet_feature_extractor(image_tensor)
    return fingerprint.squeeze().cpu().numpy()
def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f: data = json.load(f)
        return {cow_id: np.array(fp) for cow_id, fp in data.items()}
    return {}
def save_database(db_data):
    data_to_save = {cow_id: fp.tolist() for cow_id, fp in db_data.items()}
    with open(DB_PATH, 'w') as f: json.dump(data_to_save, f, indent=4)
def find_best_match(new_fingerprint, database, threshold=0.85):
    best_match_id = "Unknown Cow"; highest_similarity = -1
    for cow_id, saved_fingerprint in database.items():
        similarity = 1 - cosine(new_fingerprint, saved_fingerprint)
        if similarity > highest_similarity:
            highest_similarity = similarity
            if similarity >= threshold: best_match_id = cow_id
    return best_match_id, highest_similarity
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8'); plt.close(fig)
    return img_str
def classic_bead_arrangement(image_path):
    image = cv2.imread(image_path); gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 40, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bead_locations = []; vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0: cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"]); bead_locations.append((cX, cY)); cv2.drawContours(vis_image, [c], -1, (0, 255, 0), 2); cv2.circle(vis_image, (cX, cY), 5, (0, 0, 255), -1)
    avg_dist = 0
    if len(bead_locations) > 1:
        total_dist = sum(np.sqrt((bead_locations[i+1][0] - bead_locations[i][0])**2 + (bead_locations[i+1][1] - bead_locations[i][1])**2) for i in range(len(bead_locations) - 1))
        avg_dist = total_dist / (len(bead_locations) - 1)
        for i in range(len(bead_locations) - 1): cv2.line(vis_image, bead_locations[i], bead_locations[i+1], (255, 0, 0), 2)
    mapped_value = (avg_dist / 100.0) * 9.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap='gray'); axes[0].set_title('Original Grayscale'); axes[0].axis('off')
    axes[1].imshow(edges, cmap='gray'); axes[1].set_title('Edge Detection'); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)); axes[2].set_title(f'Bead Centroids (Avg Dist: {avg_dist:.2f})'); axes[2].axis('off')
    return int(mapped_value), fig_to_base64(fig)
def classic_bead_density(image_gray):
    binary = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    num_labels, labeled = cv2.connectedComponents(binary)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_gray, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(binary, cmap='gray'); axes[1].set_title('Thresholded'); axes[1].axis('off')
    axes[2].imshow(labeled, cmap='viridis'); axes[2].set_title(f'Connected Components: {num_labels}'); axes[2].axis('off')
    density = num_labels / (image_gray.size)
    return round(density * 10e5), fig_to_base64(fig)
def classic_ridge_orientation(image_gray):
    roi = image_gray[:, :image_gray.shape[1] // 2]
    lbp = local_binary_pattern(roi, 8, 1, method='uniform')
    gx = cv2.Sobel(lbp, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(lbp, cv2.CV_64F, 0, 1, ksize=3)
    orientation_deg = (np.degrees(0.5 * np.arctan2(2 * (gy * gx), (gx*gx) - (gy*gy))) + 180) % 180
    hist, bin_edges = np.histogram(orientation_deg, bins=16, range=[0, 180])
    consensus = bin_edges[np.argmax(hist)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_gray, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(roi, cmap='gray'); axes[1].set_title('ROI'); axes[1].axis('off')
    axes[2].imshow(lbp, cmap='gray'); axes[2].set_title('LBP Image'); axes[2].axis('off')
    axes[3].bar(bin_edges[:-1], hist, width=180/16 * 0.9, align='edge'); axes[3].axvline(consensus, color='r', linestyle='--'); axes[3].set_title('Orientation Histogram')
    return float(consensus), fig_to_base64(fig)
def classic_ridge_density(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    valid_contours = [c for c in contours if cv2.arcLength(c, False) > 20]
    cv2.drawContours(vis_image, valid_contours, -1, (0, 255, 0), 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(blurred, cmap='gray'); axes[0].set_title('Blurred Image'); axes[0].axis('off')
    axes[1].imshow(edges, cmap='gray'); axes[1].set_title('Edge Detection'); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)); axes[2].set_title('Ridge Contours'); axes[2].axis('off')
    total_length = sum(cv2.arcLength(c, False) for c in valid_contours)
    density = total_length / gray.size
    return round(density * 10e4), fig_to_base64(fig)
def classic_average_bead_size(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    beads = [c for c in contours if 50 < cv2.contourArea(c) < 5000]
    avg_size = 0; bead_sizes = []
    if beads: bead_sizes = [cv2.contourArea(b) for b in beads]; avg_size = np.mean(bead_sizes)
    fig = plt.figure(figsize=(10, 6)); plt.bar(range(len(bead_sizes)), bead_sizes); plt.axhline(y=avg_size, color='r', linestyle='--', label=f'Avg: {avg_size:.2f}')
    plt.title('Bead Size Distribution'); plt.xlabel('Bead Index'); plt.ylabel('Bead Area'); plt.legend()
    return round(avg_size, 2), fig_to_base64(fig)
def classic_inter_ridge_distance(image_path):
    image = cv2.imread(image_path, 0)
    _, threshed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(threshed, cv2.DIST_L2, 5)
    avg_dist = np.mean(dist[dist > 0]) if np.any(dist > 0) else 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(threshed, cmap='gray'); axes[1].set_title('Thresholded'); axes[1].axis('off')
    axes[2].imshow(dist, cmap='hot'); axes[2].set_title(f'Distance Transform (Avg: {avg_dist:.2f})'); axes[2].axis('off')
    return round(avg_dist, 2), fig_to_base64(fig)

# --- THIS IS THE MAIN CHANGE ---
def perform_classical_analysis(image_path):
    image_bgr = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hessian_gray_uint8 = (hessian(image_gray) * 255).astype(np.uint8)
    image_rgb_display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb_display); buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    features = {}; visualizations = {}; analysis_visualizations = {}
    
    fig_hessian, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_rgb_display); axes[0].set_title('Original BGR'); axes[0].axis('off')
    axes[1].imshow(image_gray, cmap='gray'); axes[1].set_title('Original Grayscale'); axes[1].axis('off')
    axes[2].imshow(hessian(image_bgr) * 255, cmap='gray'); axes[2].set_title('Hessian BGR'); axes[2].axis('off')
    axes[3].imshow(hessian_gray_uint8, cmap='gray'); axes[3].set_title('Hessian Grayscale'); axes[3].axis('off')
    visualizations['Input Images'] = fig_to_base64(fig_hessian)
    
    features['Bead Arrangement'], analysis_visualizations['1. Bead Arrangement'] = classic_bead_arrangement(image_path)
    features['Bead Density'], analysis_visualizations['2. Bead Density'] = classic_bead_density(hessian_gray_uint8)
    features['Ridge Orientation'], analysis_visualizations['3. Ridge Orientation'] = classic_ridge_orientation(image_gray)
    features['Ridge Density'], analysis_visualizations['4. Ridge Density'] = classic_ridge_density(image_bgr)
    features['Average Bead Size'], analysis_visualizations['5. Bead Size Distribution'] = classic_average_bead_size(image_bgr)
    features['Inter-Ridge Distance'], analysis_visualizations['6. Inter-Ridge Distance'] = classic_inter_ridge_distance(image_path)

    # Use Plotly for an interactive summary graph
    df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
    fig_summary = px.bar(df, x='Feature', y='Value', title='Feature Extraction Summary',
                         color='Feature', template='plotly_white')
    fig_summary.update_layout(showlegend=False, xaxis_title="", yaxis_title="Measured Value")
    summary_graph_html = fig_summary.to_html(full_html=False, include_plotlyjs='cdn')

    return original_b64, visualizations, analysis_visualizations, summary_graph_html

# --- (The Flask routes are updated to handle the new return value) ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/welcome')
def welcome(): return render_template('welcome.html')
@app.route('/how-it-works')
def how_it_works(): return render_template('how-it-works.html')
@app.route('/about')
def about(): return render_template('about.html')
@app.route('/pricing')
def pricing(): return render_template('pricing.html')
@app.route('/analysis', methods=['GET', 'POST'])
def analysis_page():
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename: return render_template('analysis_form.html')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            original_b64, visualizations, analysis_visualizations, summary_graph_html = perform_classical_analysis(filepath)
            return render_template('analysis_result.html', 
                                   original_image_b64=original_b64, 
                                   visualizations=visualizations, 
                                   analysis_visualizations=analysis_visualizations,
                                   summary_graph_html=summary_graph_html)
    return render_template('analysis_form.html')
@app.route('/enroll', methods=['GET', 'POST'])
def enroll_page():
    if request.method == 'POST':
        cow_id = request.form['cow_id']
        if not cow_id: return render_template('enroll.html', error="Cow ID is required.")
        if 'file' not in request.files or not request.files['file'].filename: return render_template('enroll.html', error="Image file is required.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            image_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            mask = predict_muzzle_mask(image_rgb); fingerprint = extract_fingerprint(image_rgb, mask)
            database = load_database(); database[cow_id] = fingerprint; save_database(database)
            return render_template('enroll.html', success=f"Successfully enrolled cow: {cow_id}")
    return render_template('enroll.html')
@app.route('/identify', methods=['GET', 'POST'])
def identify_page():
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename: return render_template('identify.html', error="Image file is required.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            database = load_database()
            if not database: return render_template('identify.html', error="Database is empty. Enroll a cow first.")
            image_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            mask = predict_muzzle_mask(image_rgb); unknown_fingerprint = extract_fingerprint(image_rgb, mask)
            match_id, score = find_best_match(unknown_fingerprint, database)
            result = f"Best Match: {match_id} (Similarity: {score:.2%})"
            return render_template('identify.html', result=result)
    return render_template('identify.html')
