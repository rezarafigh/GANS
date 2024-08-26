from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from werkzeug.utils import secure_filename
from src.generator import Generator
from src.utils import create_noise, denormalize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load pre-trained generator
generator = Generator()
generator.load_state_dict(torch.load('path_to_trained_model.pth'))
generator.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Generate an image from the uploaded file
            noise = create_noise(1, 100, 'cpu')  # Adjust size and device if necessary
            generated_image = generator(noise)
            # Convert tensor to image file
            generated_image = denormalize(generated_image)
            save_path = 'path_to_save_generated_image'
            torchvision.utils.save_image(generated_image, save_path)
            return render_template('index.html', user_image=file_path, generated_image=save_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
