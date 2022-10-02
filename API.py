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
    res_art = requests.get("https://openaccess-api.clevelandart.org/api/artworks/")
    response_art = json.loads(res_art.text)

    art_url = str(response_art["data"][random.randint(0, 999)]["images"]["web"]["url"])
    print(art_url)

    return art_url


@app.route('/generateImage', methods=['GET'])
def generate_image():
    searchword = request.args.get('key', '')
    url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"

    res = requests.get(url)
    response = json.loads(res.text)

    res_art = requests.get("https://openaccess-api.clevelandart.org/api/artworks/")
    response_art = json.loads(res_art.text)
    art_url = str(response_art["data"][random.randint(0, 999)]["images"]["web"]["url"])

    if len(response["collection"]["items"]) == 0:
        searchword = "space"
        url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"
        res = requests.get(url)
        response = json.loads(res.text)

    index = random.randint(0, len(response["collection"]["items"])) - 1
    imagen_url = str(response["collection"]["items"][index]["links"][0]["href"])

    m = Modelo.ArtModel(imagen_url, art_url)
    m.download_image(m.url, "C:/Users/Usuario/Desktop/spaceapps/imagenes/", m.file_name)
    m.download_image(m.art_url, "C:/Users/Usuario/Desktop/spaceapps/imagenes/", "Arte")

    m.content = m.load_file(m.content_path)
    m.style = m.load_file(m.style_path)
    m.im = m.img_preprocess(m.content_path)
    m.im2 = m.deprocess_img(m.im)
    m.model.summary()
    m.model = m.get_model()
    m.model.summary()
    m.model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    m.best, m.best_loss, m.image = m.run_style_transfer(m.content_path, m.style_path, epochs=2500)

    with open("imagenes/1.jpg", "rb") as image2string:
        converted_string = base64.b64encode(image2string.read())
    print(converted_string)

    return converted_string


app.run(debug=True)
