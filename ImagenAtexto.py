import base64

with open("food.jpeg", "rb") as image2string:
    converted_string = base64.b64encode(image2string.read())
print(converted_string)

with open('encode.bin', "wb") as file:
    file.write(converted_string)

