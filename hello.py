import json
import random
import requests
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/generateImage', methods=['GET'])
def generate_image():
    searchword = request.args.get('key', '')
    url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"

    res = requests.get(url)
    response = json.loads(res.text)

    if len(response["collection"]["items"]) == 0:
        searchword = "space"
        url = "https://images-api.nasa.gov/search?q=" + searchword + "&media_type=image"
        res = requests.get(url)
        response = json.loads(res.text)

    indice = random.randint(0, len(response["collection"]["items"])) - 1
    print(str(response["collection"]["items"][indice]["links"][0]["href"]))

    return str(response["collection"]["items"][indice]["links"][0]["href"])


app.run(debug=True)
