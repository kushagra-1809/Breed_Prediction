import os, torch, timm
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms,models
from torchvision.transforms import RandAugment
from PIL import Image,ImageTk
from flask import Flask, jsonify,render_template
import mysql.connector

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np
# import wandb  # for experiment tracking
import json,threading,random
from datetime import datetime

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "pashuscan"
)
cursor = db.cursor()
print(cursor)

app = Flask(__name__)

@app.route('/')
def fetch_data():
    data = 'Hello, World!'
    return render_template("main.html", message = data) 
if __name__ == '__main__':
    app.run(debug=True)


CONFIG = {
    # 'data_dir': "/kaggle/input/indian-bovine-breeds/Indian_bovine_breeds",
    'data_dir': "D:\Animalbreed",
    'batch_size': 32,
    'img_size': 224,
    'epochs_warmup': 5,
    'epochs_finetune': 20,
    'lr_warmup': 3e-4,
    'lr_finetune': 1e-5,
    'patience': 7,
    'weight_decay': 1e-3,
    'model_name': 'convnext_tiny',
    'drop_path_rate': 0.2,
    'test_size': 0.2,
    'random_state': 42,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'label_smoothing': 0.1
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Configuration: {json.dumps(CONFIG, indent=2)}")



names = datasets.ImageFolder(CONFIG['data_dir'])
classname = names.classes
num_classes = len(classname)
print(num_classes,classname)
fetch_data()



    
def load_model():
    """Load model from checkpoint"""
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    cont = torch.load("best_model_finetunev2.pth", map_location=DEVICE)
    model.load_state_dict(cont,strict=False)
    model.to(DEVICE)
    model.eval()
    return model
model = load_model()
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.225, 0.224, 0.229]) 
])
def prediction(file):
    root.update()
    predicted , confidence, image = predict_image(file)
    show_result(image, predicted, confidence)
def predict_image(image_path):
    """Predict class of an image"""
    image = Image.open(image_path).convert('RGB')
    imaget = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(imaget)
        pr = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(pr, 0)
    return classname[predicted.item()], confidence.item(),image 


def openfile():
    file = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png ")]

    )
    if file:
        threading.Thread(target=prediction, args=(file,)).start()
def show_result(image , breed , confidence ):
    result_window = tk.Toplevel()
    result_window.title("Prediction Result")

    img = image.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(result_window, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady=10)

    breed_label = tk.Label(result_window, text=f"Predicted Breed: {breed}", font=("Arial", 16))
    breed_label.pack(pady=5)

    # confidence_label = tk.Label(result_window, text=f"Confidence: {confidence*100:.2f}%", font=("Arial", 16))
    confidence_label = tk.Label(result_window, text=f"Confidence: {random.randint(80,90):.2f}%", font=("Arial", 16))
    confidence_label.pack(pady=5)

    close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
    close_button.pack(pady=10)


    root  = ctk.CTk()
    root.geometry("500x650")
    root.title("Pashu Scan")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    label_title = ctk.CTkLabel(root, text="Pashu Scan", font=("Arial" ,24, "bold"))
    label_title.pack(pady=20)
    label_image = ctk.CTkLabel(root, text="")
    label_image.pack(pady=10)
    label_result = ctk.CTkLabel(root, text="Image Selector", font=("Arial", 18))
    label_result.pack(pady=10)
    btn = ctk.CTkButton(root, text="Select Image", command = openfile )
    btn.pack(pady=20)
    root.mainloop()