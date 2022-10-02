import json
import random
import requests
import tensorflow as tf
import Modelo as Modelo
import base64
from flask import Flask, request

app = Flask(__name__)


@app.route('/generateImage', methods=['GET'])
def generate_image():
    searchword = request.args.get('key', '')
    url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"

    res = requests.get(url)
    response = json.loads(res.text)

    res_art = requests.get("https://openaccess-api.clevelandart.org/api/artworks/")
    response_art = json.loads(res_art.text)
    # art_url = str(response_art["data"][random.randint(0, 999)]["images"]["web"]["url"])

    art_url="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpaperset.com%2Fw%2Ffull%2F2%2F2%2F9%2F207342.jpg&f=1&nofb=1&ipt=2fda1fa6e3b7fd801376864f512323acd52519acb2711b81b5fcc18ae8889151&ipo=images"
    if len(response["collection"]["items"]) == 0:
        searchword = "space"
        url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"
        res = requests.get(url)
        response = json.loads(res.text)

    index = random.randint(0, len(response["collection"]["items"])) - 1
    imagen_url = str(response["collection"]["items"][index]["links"][0]["href"])

    print("PROCESANDO REQUEST")
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
