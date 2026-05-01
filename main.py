from src.data import load_data
from src.preprocess import preprocess, flatten, add_noise
from src.visualize import show_images
from src.model import Autoencoder

train, test = load_data()

print("Train size:", len(train))
print("Test size:", len(test))

# load
train, test = load_data()

# preprocess
train_loader, test_loader = preprocess(train, test)

# sample batch
images, labels = next(iter(train_loader))

# show original
show_images(images, "Original Images")

# flatten
flat = flatten(images)
print("Flatten shape:", flat.shape)

# add noise
noisy = add_noise(images)

# show noisy
show_images(noisy, "Noisy Images")

model = Autoencoder()
print(model)

from src.train import train_model

# train model
model, losses = train_model(model, train_loader)

from src.visualize import show_reconstruction

# visualize reconstruction
show_reconstruction(model, test_loader)

from src.model import ConvAutoencoder
from src.train import train_conv_model
from src.visualize import show_reconstruction

# Conv model
conv_model = ConvAutoencoder()

# train
conv_model = train_conv_model(conv_model, train_loader)

# visualize
show_reconstruction(conv_model, test_loader)

from src.train import train_denoising
from src.visualize import show_denoising

# train denoising model
conv_model = train_denoising(conv_model, train_loader)

# visualize
show_denoising(conv_model, test_loader)