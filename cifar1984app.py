
from flask import Flask, render_template, request, redirect
import requests
from werkzeug.utils import secure_filename
import os,os.path
from PIL import Image
import os, os.path
import torch

from Net import Net
from img_to_tensor import img_to_tensor
from predictor import predictor
#IMG_TO_TENSOR

#PREDICTOR

app = Flask(__name__)
net = Net(feat_1=12,feat_2=24,feat_3=36)
net.load_state_dict(torch.load('new_model_varb.pth'))


upload_folder = os.path.join('static', 'uploads')

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET','POST'])
def root():
    if request.method == 'GET':
        return render_template('index-online.html', data=[None])
    if request.method == 'POST':
        return render_template('index-online.html', data=["No image received!"])

@app.route('/predict.html', methods=['GET','POST'])
def pred():
    #try:
   #     data_maker=data_inh
    #except:
   #     data_maker= [ 0 for i in range(11)]

    #if request.method == 'GET':
    #    print(data_maker)
    #    img_url="https://pbs.twimg.com/media/F3ml1hsa4AAA2d1?format=jpg&name=large"
    #    return render_template('predict.html', data=data_maker, img=img_url)

    if request.method == 'POST':
        try:

            img_up=request.files['Upload Image']

            filename = secure_filename(img_up.filename)

            img_up.save(os.path.join(app.config['UPLOAD'], filename))

            img_url = os.path.join(app.config['UPLOAD'], filename)
            img_raw=Image.open(img_up)
            print("img_raw type= %s"%type(img_raw))
            width, height = img_raw.size
        except:
            try:
                img_url = request.form['content']
                img_raw = Image.open(requests.get(img_url, stream=True).raw)
                width, height = img_raw.size
                print(type(img_raw))
                print(img_raw)
            except:
                img_raw = None
                print("no img")
                return redirect('/')
        #print(img_url)
        #ztimage=Image.open(requests.get(img_url, stream=True).raw)
        #ztimage=img_ur
        if height>=width: data=[img_raw,width/height*400,400]      #width, height
        else: data=[img_raw,400,height/width*400]

        rrat,wp,hp=img_to_tensor(img_obj=img_raw,obj_type='raw', threshold = 6/4, display=False)
        print(wp)
        print(hp)
        print(rrat.size())

        sortt=predictor(model=net, img_tensor=rrat)
        print(sortt)
        for j in range(6):
            data.append(sortt[j])
        data.append(wp)
        data.append(hp)
        data.append(min(data[1],data[2]))

    return render_template('predict.html', data=data, img=img_url)

if __name__ == '__main__':

    app.run()#debug=True, use_reloader=False
