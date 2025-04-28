import os
pip install Flask
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Ensure there is no space before the file path and use raw string literal
disease_info = pd.read_csv(r'C:\Users\chaka\Downloads\Plant-Disease-Detection-main-main\Plant-Disease-Detection-main-main\Flask Deployed App\disease_info.csv', encoding='cp1252')

# Corrected file path: no space at the beginning, using raw string literal
supplement_info = pd.read_csv(r'C:\Users\chaka\Downloads\Plant-Disease-Detection-main-main\Plant-Disease-Detection-main-main\Flask Deployed App\supplement_info.csv', encoding='cp1252')

# Load the model
model = CNN.CNN(39)
model.load_state_dict(torch.load(r'C:\Users\chaka\Downloads\Plant-Disease-Detection-main-main\Plant-Disease-Detection-main-main\plant_disease_model_1_latest.pt'))

model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask app
app = Flask(__name__, template_folder=r'C:\Users\chaka\Downloads\Plant-Disease-Detection-main-main\Plant-Disease-Detection-main-main\Flask Deployed App\templates')

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(r'C:\Users\chaka\Downloads\Plant-Disease-Detection-main-main\Plant-Disease-Detection-main-main\Flask Deployed App\static\uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
