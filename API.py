import json
import random
import requests
import base64
from io import BytesIO
from Modelo import Model
from flask import Flask, request

m = Model()

app = Flask(__name__)


@app.route('/generateImage', methods=['GET'])
def generate_image():
    search_word = request.args.get('key', '')
    url = "https://images-api.nasa.gov/search?q=" + search_word + "&media_type=image"

    res = requests.get(url)
    response = json.loads(res.text)

    art_url = "Art/" + str(random.randint(1, 26)) + ".jpg"

    if len(response["collection"]["items"]) == 0:
        search_word = "space"
        url = "https://images-api.nasa.gov/search?q=" + search_word + "&media_type=image"
        res = requests.get(url)
        response = json.loads(res.text)

    index = random.randint(0, len(response["collection"]["items"])) - 1
    imagen_url = str(response["collection"]["items"][index]["links"][0]["href"])

    #print('Content: {content}'.format(content=imagen_url))
    #print('Style: {style}'.format(style=art_url))

    image = m.generate_image(str(random.randint(0, 999)) + ".jpg",
                             imagen_url,
                             art_url)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return str(img_str)


app.run(debug=False)
