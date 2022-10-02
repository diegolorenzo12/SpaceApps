import tensorflow as tf
import numpy as np
import tensorflow.keras.preprocessing.image as process_im
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import functools
import IPython.display
import urllib.request

#path = 'https://images-assets.nasa.gov/image/PIA11421/PIA11421~thumb.jpg'
path = 'https://images-assets.nasa.gov/image/NHQ201906010010/NHQ201906010010~thumb.jpg'
#path = "https://images-assets.nasa.gov/image/NHQ201906010015/NHQ201906010015~thumb.jpg"

def download_image(url, file_path, file_name):
    full_path = file_path + file_name + '.jpg'
    urllib.request.urlretrieve(url, full_path)


url = path
file_name = "content"
download_image(url, 'C:/Users/Usuario/Desktop/spaceapps/imagenes/', file_name)

content_path = 'imagenes/content.jpg'
style_path = 'Arte.png'


def load_file(image_path):
    image = Image.open(image_path)
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.ANTIALIAS)
    im_array = process_im.img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    return im_array


def show_im(img, title=None):
    img = np.squeeze(img, axis=0)
    plt.imshow(np.uint8(img))
    if title is None:
        pass
    else:
        plt.title(title)
    plt.imshow(np.uint8(img))


content = load_file(content_path)
style = load_file(style_path)

plt.figure(figsize=(10, 10))
content = load_file(content_path)
style = load_file(style_path)
plt.subplot(1, 2, 1)
show_im(content, 'Content Image')
plt.subplot(1, 2, 2)
show_im(style, 'Style Image')


# plt.show()


# %%103.939
def img_preprocess(img_path):
    images = load_file(img_path)
    img = tf.keras.applications.vgg19.preprocess_input(images)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3

    x[:, :, 0] += 105
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


im = img_preprocess(content_path)
im_2 = deprocess_img(im)

content_layers = ['block1_conv2',
                  'block2_conv2',
                  'block3_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
number_content = len(content_layers)
number_style = len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = style_output + content_output
    return models.Model(vgg.input, model_output)


model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
model.summary()

model = get_model()
model.summary()


# %%
def get_content_loss(noise, target):
    loss = tf.reduce_mean(tf.square(noise - target))
    return loss


def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    n = tf.shape(vector)[0]
    gram_matrix = tf.matmul(vector, vector, transpose_a=True)
    return gram_matrix / tf.cast(n, tf.float32)


def get_style_loss(noise, target):
    gram_noise = gram_matrix(noise)
    # gram_target=gram_matrix(target)
    loss = tf.reduce_mean(tf.square(target - gram_noise))
    return loss


def get_features(model, content_path, style_path):
    content_img = img_preprocess(content_path)
    style_image = img_preprocess(style_path)

    content_output = model(content_img)
    style_output = model(style_image)

    content_feature = [layer[0] for layer in content_output[number_style:]]
    style_feature = [layer[0] for layer in style_output[:number_style]]
    return content_feature, style_feature


def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    output = model(image)
    content_loss = 0
    style_loss = 0

    noise_style_features = output[:number_style]
    noise_content_feature = output[number_style:]

    weight_per_layer = 1.0 / float(number_style)
    for a, b in zip(gram_style_features, noise_style_features):
        style_loss += weight_per_layer * get_style_loss(b[0], a)

    weight_per_layer = 1.0 / float(number_content)
    for a, b in zip(noise_content_feature, content_features):
        content_loss += weight_per_layer * get_content_loss(a[0], b)

    style_loss *= style_weight
    content_loss *= content_weight

    total_loss = content_loss + style_loss
    return total_loss, style_loss, content_loss


def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dictionary)

    total_loss = all_loss[0]
    return tape.gradient(total_loss, dictionary['image']), all_loss


model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')


# %%
def run_style_transfer(content_path, style_path, epochs=500, content_weight=1e3, style_weight=1e1):
    model = get_model()

    for layer in model.layers:
        layer.trainable = False

    content_feature, style_feature = get_features(model, content_path, style_path)
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]

    noise = img_preprocess(content_path)
    noise = tf.Variable(noise, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    dictionary = {'model': model,
                  'loss_weights': loss_weights,
                  'image': noise,
                  'gram_style_features': style_gram_matrix,
                  'content_features': content_feature}
    norm_means = np.array([105, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    image_no = 1
    imgs = []
    for i in range(epochs):
        grad, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grad, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())
        if i % 25 == 0:
            plot_img = noise.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            m = Image.fromarray(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            name = r'imagenes/' + str(image_no) + '.jpg'
            m.save(name, 'JPEG')
            image_no += 1
    IPython.display.clear_output(wait=True)
    return best_img, best_loss, imgs

im = Image.open(r"imagenes/1.jpg")
im.show()
best, best_loss, image = run_style_transfer(content_path, style_path, epochs=2500)
