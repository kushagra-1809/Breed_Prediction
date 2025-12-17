import os, torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms,models
from torchvision.transforms import RandAugment
from PIL import Image
from flask import Flask, jsonify,render_template,request,redirect,url_for
import mysql.connector
from werkzeug.utils import secure_filename
import json,threading,random
from datetime import datetime

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
def dbconnection():
    return mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "pashuscan")
cursor = dbconnection().cursor()
print(cursor)
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


    
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
cont = torch.load("best_model_finetunev2.pth", map_location=DEVICE)
model.load_state_dict(cont,strict=False)
model.to(DEVICE)
model.eval()
    

 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.225, 0.224, 0.229]) 
])





def gentagid():
    p = "1100"
    r = "".join([str(random.randint(0,9))for _ in range(8) ])
    return p + r

def predict_image(image_path):
    """Predict class of an image"""
    image = Image.open(image_path).convert('RGB')
    imaget = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(imaget)
        pr = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(pr, 0)
        
    return classname[predicted.item()], confidence.item()


app = Flask(__name__)
@app.route('/')
def dashboard():
    return render_template("main.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    tagid = gentagid()
    if request.method == 'POST':
        
        image_file = request.files['image']

        filename = secure_filename(f"{tagid}.png")
        save_path =  os.path.join('static/images', filename)
        image_file.save(save_path)

        breed, confidence = predict_image(save_path)
        if breed != "":
            db = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "pashuscan")
        cursor = db.cursor()
        cursor.execute("SELECT acc_no FROM breeds WHERE breed = %s", (breed,))
        result = cursor.fetchone()
        acc_no = result[0] if result else ""
        print(acc_no)
        cursor.close()

        return render_template("save.html",
                               tagid=tagid,
                               photo=filename,
                               breed=breed,
                                acc_no=acc_no,
                               confidence = round(confidence*100,2))
    return render_template("register.html", tagid=tagid)

@app.route('/save', methods=['POST'])
def save():
    try:
        tagid = request.form.get('tagid')
        photo = request.form.get('photo')  
        breed = request.form.get('breed')
        acc_no = request.form.get('acc_no', '')
        cross_breed = request.form.get('cross_breed', '')
        owner_name = request.form.get('owner_name')
        contact = request.form.get('contact')
        address = request.form.get('address')
        age = request.form.get('age')
        vacc = request.form.get('vacc')
        calvings = request.form.get('calvings')
        ai = request.form.get('ai')
        pd = request.form.get('pd')

        if not tagid or not owner_name:
            return "Missing required fields", 400

        photo_path = os.path.join("static/uploads", photo) if photo else ""
        db = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "pashuscan")
        cursor = db.cursor()
        
        sql = """
            INSERT INTO animals
            (acc_no, tag_id, breed, owner_name, contact, address, dob, vacc, calving, pd, img, ai, cross_breed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (acc_no, tagid, breed, owner_name, contact, address, age, vacc, calvings, pd, photo_path, ai, cross_breed))
        db.commit()
        cursor.close()
        db.close()

        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Error saving animal: {e}", 500

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    db = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "pashuscan")
    cursor = db.cursor(dictionary=True)

    if request.method == "POST":
        tag_id = request.form.get('tag_id')
        cursor.execute("SELECT * FROM animals WHERE tag_id = %s", (tag_id,))
        result = cursor.fetchone()
        return render_template("details.html",tag_id= tag_id,owner_name = result['owner_name'],breed = result['breed'],acc = result['acc_no'],dob = result['dob'],vacc = result['vacc'],cross = result['cross_breed'], calving = result['calving'],contact = result['contact'],address = result['address'],photo = result['img'],ai = result['ai'],pd = result['pd'])
        
    return render_template("scan.html")

@app.route('/details/<tag_id>')
def details(tag_id):
     db = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "pashuscan")
     cursor = db.cursor(dictionary=True)
     cursor.execute("SELECT * FROM animals WHERE tag_id = %s", (tag_id,))
     result = cursor.fetchone()
     return render_template("details.html",tag_id= tag_id,owner_name = result['owner_name'],breed = result['breed'],acc = result['acc_no'],dob = result['dob'],vacc = result['vacc'],cross = result['cross_breed'], calving = result['calving'],contact = result['contact'],address = result['address'],photo = result['img'],ai = result['ai'],pd = result['pd'])

if __name__ == '__main__':
    app.run(debug=True)







