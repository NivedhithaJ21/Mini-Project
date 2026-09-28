from flask import Flask, render_template, request
from PIL import Image, ImageStat
import numpy as np
import io

app = Flask(__name__)

def analyze_glaucoma(img):
    img = img.convert('RGB').resize((512, 512))
    arr = np.array(img)
    gray = np.array(img.convert('L'))

    red = arr[:,:,0].astype(float)
    green = arr[:,:,1].astype(float)
    blue = arr[:,:,2].astype(float)

    h, w = gray.shape
    cx, cy = h//2, w//2

    # Central region (optic disc area)
    center = gray[cx-80:cx+80, cy-80:cy+80]
    outer = np.concatenate([gray[:cx-80,:].flatten(), gray[cx+80:,:].flatten()])

    center_mean = center.mean()
    outer_mean = outer.mean()
    center_brightness_ratio = center_mean / (outer_mean + 1)

    # Color features
    red_mean = red.mean()
    green_mean = green.mean()
    blue_mean = blue.mean()

    # Contrast
    contrast = float(gray.std())

    # Bright pixel ratio (large optic cup indicator)
    bright_pixels = (gray > 200).sum() / gray.size
    dark_pixels = (gray < 30).sum() / gray.size

    # Edge density
    from PIL import ImageFilter
    edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)
    edge_density = edge_arr.mean()

    # Risk scoring
    risk = 0

    # Cup-to-disc: bright center vs dark background
    if center_brightness_ratio > 1.8:
        risk += 3
    elif center_brightness_ratio > 1.4:
        risk += 1

    # Color: glaucoma = pale/yellowish, normal = orange-pink
    if red_mean > green_mean and red_mean > blue_mean and red_mean > 120:
        risk += 2
    if green_mean > red_mean:
        risk -= 1

    # Low contrast = damaged retina
    if contrast < 35:
        risk += 2
    elif contrast > 65:
        risk -= 1

    # Too many bright spots = large cup
    if bright_pixels > 0.25:
        risk += 2
    elif bright_pixels < 0.05:
        risk -= 1

    # Too many dark pixels = abnormal
    if dark_pixels > 0.3:
        risk += 1

    # Low edge density = less detail = possible damage
    if edge_density < 10:
        risk += 1
    elif edge_density > 25:
        risk -= 1

    prob = min(max(risk / 9.0, 0.0), 1.0)
    return prob, {
        'center_ratio': round(center_brightness_ratio, 2),
        'contrast': round(contrast, 2),
        'bright_pixels': round(bright_pixels * 100, 2),
        'edge_density': round(edge_density, 2),
        'red_mean': round(red_mean, 2),
        'green_mean': round(green_mean, 2),
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    details = None
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        prob, details = analyze_glaucoma(img)
        confidence = round(prob * 100, 2)
        print("Prob:", prob, "Details:", details)
        if prob >= 0.5:
            result = "⚠️ Glaucoma Detected - High IOP Risk!"
        elif prob >= 0.25:
            result = "🔶 Borderline - Monitor Regularly"
        else:
            result = "✅ Normal Eye - No Glaucoma Detected"
    return render_template('index.html', result=result, confidence=confidence, details=details)

if __name__ == '__main__':
    app.run(debug=True)