from datetime import datetime
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

def test_utils():
  print('Test utils ok')

def check_and_get_patch_2D(img, u, v, m):
    # Check
    if u - m < 0 or u + m >= img.shape[0]:
        return False, None
    
    # Check
    if v - m < 0 or v + m >= img.shape[1]:
        return False, None
    
    # Compute
    patch = np.empty((2*m+1, 2*m+1))
    for i in range(2*m+1):
        for j in range(2*m+1):
            patch[i, j] = img[u-m+i,v-m+j]
    return True, patch

# Fonction gaussienne 1D
def f(x, sig):
    return np.exp(-x**2/(2*sig**2))

# Fonction gaussienne 2D
def f_2D(x, y, sig):
    return f(x, sig)*f(y, sig)


# Calcul d'un filtre gaussien en 2D
def get_gauss_filter_2D(sigma, epsilon):
    # Trouver le plus grand entier L tel que f(L) > epsilon
    L =  int(sigma * np.sqrt(-2 * np.log(epsilon)))

    # Calculer les poids
    w = np.empty((2*L+1, 2*L+1))
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            w[L+i,L+j] = f_2D(i,j,sigma)

    # Normalisation obtenir un filtre de lissage
    w /= w.sum()
    return w

def apply_filter_2D(im, W):
    fil_img = np.empty(im.shape)
    
    if np.mod(W.shape[0],2) == 0 or np.mod(W.shape[1],2) == 0:
        raise ValueError("Le filtre doit avoir une longueur impaire.")
    else:
        m = int((W.shape[0] - 1) / 2)
    
    # Calculer l'image filtree
    for u in range(im.shape[0]):
        for v in range(im.shape[1]):
            ok, patch = check_and_get_patch_2D(im, u, v, m)
            if ok:
                mean_val = np.sum(patch * W)
            else:
                mean_val = 0
            fil_img[u, v] = mean_val
    return fil_img


def show_function(x, y, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot()

    # Set title if any
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])

    # Show function
    ax.plot(x, y)


def show_vectors(*vecs, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot()

    # Set title if any
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])

    # Check for labels
    if 'labels' in kwargs.keys():
        labels = kwargs['labels']
        if len(labels) != len(vecs):
            raise ValueError(f'The numbers of labels and vectors are not the same! {len(labels)} != {len(vecs)}.')

        # Show vectors with labels
        for (v, l) in zip(vecs, labels):
            plt.plot(range(len(v)), v, label=l)
        plt.legend()
    else:
        # Show vectors without labels
        for v in vecs:
            plt.plot(range(len(v)), v)


def show_image(img, check_bounds=True, **kwargs):
    # Select data source: load image or use provided one
    if type(img) == str:
        image_to_dsp = plt.imread(img)
    elif type(img) == np.ndarray:
        image_to_dsp = img
    else:
        raise ValueError("Syntaxe : Vous avez oublié d'indiquer le nom du fichier ou celui-ci n'est oas une chaine de caractères!")

    # Check image values
    if check_bounds:
        img_min = np.min(image_to_dsp)
        img_max = np.max(image_to_dsp)
        is_in_bounds = (img_min >= 0 and img_max <= 255)
        if not is_in_bounds:
            raise ValueError(f"Valeur non autorisée dans l'image! Min= {img_min}; Max= {img_max}.")

    # Create new figure, add subplot, set title if provided
    fig = plt.figure()
    ax = fig.add_subplot()

    # Set title if any
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])

    # Show image
    ax.imshow(image_to_dsp, norm=matplotlib.colors.Normalize(vmin=0, vmax=255), cmap='gray')


def load_and_show_image(file, **kwargs):
    img = plt.imread(file)
    # print(repr(img))
    # print(img.shape)
    show_image(img, **kwargs)
    return img


def show_histogram(img, cumulative=False, **kwargs):
    """
    Black-box histogram computation and display
    :param img:
    :param cumulative: if True, displays a cumulative histogram
    :return: data from the histogram
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    # Set title if any
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])

    # Compute histogram
    hist, _ = np.histogram(img.ravel(), bins=np.arange(257)-0.5)

    # Accumulate?
    if cumulative:
        hist = np.cumsum(hist)

    # Show, and return data
    ax.bar(range(256), hist, width=1.0)
    # Pb d'arrondi ! hist, _, _ = ax.hist(img.ravel(), bins=256, cumulative=cumulative)
    return hist


def show_bars(hist, **kwargs):
    """
    White-box histogram display
    :param hist:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    # Set title if any
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])

    # Show bars
    ax.bar(range(256), hist, width=1.0, color='orange')


def print_timestamp(txt=''):
    """
    Timestamping utility function
    :param txt:
    :return:
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{timestamp}] {txt}')


def LUT(val, p0, p1):
    """
    Look Up Table
    :param val: value in [0, 255]
    :param p0: start/bottom of the linear ramp, in [0, 255]
    :param p1: stop/top of the linear ramp, in [0, 255]
    :return: 0 if val <= p0, 255 if val >= p1, and linear interpolation if in between
    """
    slope = 255 / (p1 - p0)
    if val <= p0:
        return 0
    elif val >= p1:
        return 255
    else:
        # Here: p0 < val < p1
        return (val - p0) * slope


def apply_LUT(image, p0, p1):
    """
    Applique la LUT a une image
    :param image:
    :param p0:
    :param p1:
    :return:
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = LUT(image[i, j], p0, p1)


# ================================================================================================================
# Code below taken from: https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
# See also: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
def convolve2D_slow(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
# ================================================================================================================

# Filtrage

def g_sigma(x,sigma):
   y = np.exp(-0.5*x**2/(sigma**2))
   return y

def W_gauss(sigma):
   eps = 1e-3
   L = np.floor(np.sqrt(-2 * np.log(eps)) * sigma)
   G = g_sigma(np.arange(-L,L+1),sigma)
   G /= G.sum()
   return G

def W_gauss_2(sigma):
   G1 = W_gauss(sigma)[:,np.newaxis]
   G = np.dot(G1,G1.T)
   return G

# calcul de l'histogramme cumule
def hist_cumul(im):
    # calcul de l'histogramme
    classes = np.arange(-1,256)
    classes[0] = -0.1
    #classes = [-0.1] + [i for i in range(0,256)]
    hist, _ = np.histogram(im.flatten(),bins=classes)
    #hist, _, _ = plt.hist(im.flatten(),bins=classes)  #histc(classes, im, normalization=%f);
    Hist = np.cumsum(hist)

    return Hist

# renvoie une image 500x500 avec un disque blanc
# sur fond noir
def disque():
    X,Y = np.meshgrid(np.arange(500),np.arange(500))
    im = 255. * ( (X - 250) ** 2 + (Y - 250) ** 2 < 150**2)
    return im


# Fonctions issues du TP4

def norme_gradient(im):
    Dx = 0.5*np.array([[-1,0,1]])
    im_x = convolve2d(im*1.,Dx,mode='same')
    Dy = np.array([[1],[0],[-1]])
    im_y = convolve2d(im*1.,Dy,mode='same')
    imn = np.sqrt(im_x ** 2 + im_y ** 2)
    return imn


def contours_seuil(im, seuil):
    imn = norme_gradient(im)
    imc = np.zeros_like(imn)
    imc[imn > seuil] = 255
    return imc

def trouver_seuil(im, p):
    """ identifie le seuil tel qu'il y a
    un pourcentage p des pixels pour 
    lesquels la norme du gradient est 
    inferieure a ce seuil."""

    G = norme_gradient(im)
    H = hist_cumul(G)
    seuil = 1
    for i in range(H.shape[0]):
        if H[i]/H[-1] < p:
            seuil += 1
    return seuil

def contours_p(im, p):
    """ Determiner le seuil a partir d'un pourcentage p"""
    G = norme_gradient(im)
    imc = contours_seuil(im, trouver_seuil(im, p))
    return imc
    