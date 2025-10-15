##exo 7
import numpy as np
import matplotlib.pyplot as plt

def modifier_image(image, p0=0, p1=255):
   
    a = 255 / (p1 - p0)
    b = -a * p0
    
    
    image_transformee = a * image + b
    
    
    image_transformee = np.clip(image_transformee, 0, 255)
    
    return image_transformee


image = plt.imread("papillon.bmp")  

#exemple

p0 = 0
p1 = 150
image_transformee = modifier_image(image, p0, p1)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))


ax[0].imshow(image, vmin=0, vmax=255)
ax[0].set_title("Image originale")
ax[0].axis('off')


ax[1].imshow(image_transformee, vmin=0, vmax=255)
ax[1].set_title(f"Image après transformation affine\n(p0={p0}, p1={p1})")
ax[1].axis('off')

plt.show()

##exo 8
import numpy as np
import matplotlib.pyplot as plt

def modifier_image_gamma(image, gamma):
    
    image_normalized = image / 255.0  
    image_gamma = 255 * (image_normalized ** gamma)  
    image_gamma = np.clip(image_gamma, 0, 255)  
    return image_gamma


images = ['bastille.bmp', 'desert.bmp', 'manoir.bmp', 'fruits.bmp', 'papillon.bmp']


fig, ax = plt.subplots(len(images), 5, figsize=(15, 3 * len(images)))

for i, image_path in enumerate(images):
    image = plt.imread(image_path)
    ax[i, 0].imshow(image, vmin=0, vmax=255)
    ax[i, 0].set_axis_off()
    ax[i, 0].set_title('Original')
    
    #tests
    for j, gamma in enumerate([0.5, 1.0, 2.0]):
        image_gamma = modifier_image_gamma(image, gamma)
        ax[i, j + 1].imshow(image_gamma, vmin=0, vmax=255)
        ax[i, j + 1].set_axis_off()
        ax[i, j + 1].set_title(f'Gamma = {gamma}')
        
plt.tight_layout()
plt.show()
