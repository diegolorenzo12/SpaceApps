import json
import random
import requests
from flask import Flask, request
import tensorflow as tf
import ModeloConArgs as Modelo
from PIL import Image
import base64

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/generateImage', methods=['GET'])
async def generate_image():
    searchword = request.args.get('key', '')
    url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"

    res = requests.get(url)
    response = json.loads(res.text)

    if len(response["collection"]["items"]) == 0:
        searchword = "space"
        url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"
        res = requests.get(url)
        response = json.loads(res.text)

    index = random.randint(0, len(response["collection"]["items"])) - 1
    imagen_url = str(response["collection"]["items"][index]["links"][0]["href"])

    m = Modelo.ArtModel(imagen_url)
    m.download_image(m.url, "C:/Users/Usuario/Desktop/spaceapps/imagenes/", m.file_name)
    m.content = m.load_file(m.content_path)
    m.style = m.load_file(m.style_path)
    m.im = m.img_preprocess(m.content_path)
    m.im2 = m.deprocess_img(m.im)
    m.model.summary()
    m.model = m.get_model()
    m.model.summary()
    m.model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    m.best, m.best_loss, m.image = await m.run_style_transfer(m.content_path, m.style_path, epochs=2500)
    print("oli")


    with open("imagenes/1.jpg", "rb") as image2string:
        converted_string = base64.b64encode(image2string.read())
    print(converted_string)

    return converted_string


app.run(debug=True)
