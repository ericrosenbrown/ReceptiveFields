# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# We are going to be doing an activity about viewing images through different filters. These filters are similar to things that happen in the brain when the images from our eyes are registered in our brain.

# <codecell>

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import numpy as n

# <codecell>

barImg=mpimg.imread('bar.png')
#extract grey values
barImg = barImg[:,:,3]

# <markdowncell>

# We examine the effect on the following images. In the visual pathway the images can be seen as input from our eyes focusing on the center of our vision.

# <codecell>

imgplot = plt.imshow(barImg, cmap=cm.Greys_r)

# <codecell>

img=mpimg.imread('stinkbug.png') #change 'stinkbug.png' into your choice of animal:
# turtle.jpg, turtle2.jpg, zebra.png, doge.png, jaguar.png, leopard.png, mexicanhat.jpg
#extract grey values
bugImg = img[:,:,0]

# <codecell>

imgplot = plt.imshow(bugImg, cmap=cm.Greys_r)

# <markdowncell>

# Receptive field functions
# -------------------
#
# The following function will be used as a blurring filter.
# $$\phi(x,y) = \frac{1}{2\pi\sigma^2}\exp{\{-\frac{1}{2\pi\sigma^2}(x^2+ y^2)\}}$$

# <codecell>

def gaussian2D(x, y, sigma):
    return (1.0/(1*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))

"""make matrix from function"""
def receptiveFieldMatrix(func):
    h = 30
    g = zeros((h,h))
    for xi in range(0,h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            g[xi, yi] = func(x,y);
    return g

def plotFilter(fun):
    g = receptiveFieldMatrix(fun)
    plt.imshow(g, cmap=cm.Greys_r)

# <markdowncell>

# The function is circular symmetric, meaning it is doing the same thing around a circle.
#
# This filter cancels out higher frequencies, thus blurring the image.

# <codecell>

plotFilter(lambda x,y:gaussian2D(x,y,4))

# <markdowncell>

# Convolution is the process of applying the filter to the input image.
# $$\int \int I(x',y')\phi(x-x',y-y')dx'dy'$$
#
# When applying this filter, the result of the convolution can be visualized in an image.

# <codecell>

Img_barGaussian = signal.convolve(barImg,receptiveFieldMatrix(lambda x,y: gaussian2D(x,y,5)), mode='same')

imgplot = plt.imshow(Img_barGaussian, cmap=cm.Greys_r)

# <codecell>

Img_bugGaussian = signal.convolve(bugImg,receptiveFieldMatrix(lambda x,y: gaussian2D(x,y,3)), mode='same')

imgplot = plt.imshow(Img_bugGaussian, cmap=cm.Greys_r)

# <markdowncell>

# Difference of Gaussians
# ---------------------
#
# The mexican hat function is a difference between two of the function above, which leads to a filter that happens in certain cells in your eye. It can be seen as a basic edge detector.

# <codecell>

def mexicanHat(x,y,sigma1,sigma2):
    return gaussian2D(x,y,sigma1) - gaussian2D(x,y,sigma2)

plotFilter(lambda x,y: mexicanHat(x,y,3,4))

# <codecell>

Img_barHat = signal.convolve(barImg,receptiveFieldMatrix(lambda x,y:mexicanHat(x,y,3,4)), mode='same')

imgplot = plt.imshow(Img_barHat, cmap=cm.Greys_r)

# <codecell>

Img_bugHat = signal.convolve(bugImg,receptiveFieldMatrix(lambda x,y: mexicanHat(x,y,2,3)), mode='same')

imgplot = plt.imshow(Img_bugHat, cmap=cm.Greys_r)

# <markdowncell>

# Gabor functions
# ---------------
#
# Gabor functions are used to detect edges with a specific orientation in images. There are parts in the brain that see an image through these gabor functions and are found throughout a part of your eye.
#
# There are two different types of gabor function:
# $$g_s(x):=sin(\omega_x x + \omega_y y)\exp{\{-\frac{x^2+y^2}{2\sigma^2}\}}$$
# $$g_c(x):=cos(\omega_x x + \omega_y y)\exp{\{-\frac{x^2+y^2}{2\sigma^2}\}}$$
#

# <codecell>

def oddGabor2D(x,y,sigma,orientation):
    return math.sin(x + orientation*y) * math.exp(-(x**2 + y**2)/(2*sigma))

def evenGabor2D(x,y, sigma, orientation):
    return math.cos(x + orientation*y) * math.exp(-(x**2 + y**2)/(2*sigma))

plotFilter(lambda x,y: oddGabor2D(x,y,7,1))

# <codecell>

Img_barOddGabor = signal.convolve(barImg,receptiveFieldMatrix(lambda x,y: oddGabor2D(x,y,5,1)), mode='same')

imgplot = plt.imshow(Img_barOddGabor, cmap=cm.Greys_r)

# <codecell>

Img_bugOddGabor = signal.convolve(bugImg,receptiveFieldMatrix(lambda x,y: oddGabor2D(x,y,5,1)), mode='same')

# <markdowncell>

# In the following image one can see the edge orientations appear in the part of the eye.

# <codecell>

imgplot = plt.imshow(Img_bugOddGabor, cmap=cm.Greys_r)

# <markdowncell>

# Using the previous filter (the edge defining one) as an input to the gabor we obtain different results.

# <codecell>

Img_bugOddGaborEdge = signal.convolve(Img_bugHat,receptiveFieldMatrix(lambda x,y: oddGabor2D(x,y,5,1)), mode='same')

imgplot = plt.imshow(Img_bugOddGaborEdge, cmap=cm.Greys_r)


# <markdowncell>
# Here is an example of the other gabor filter
# <codecell>

plotFilter(lambda x,y: evenGabor2D(x,y,7,1))

Img_barEvenGabor = signal.convolve(barImg,receptiveFieldMatrix(lambda x,y: evenGabor2D(x,y,5,1)), mode='same')

imgplot = plt.imshow(Img_barEvenGabor, cmap=cm.Greys_r)

# <codecell>

Img_bugEvenGabor = signal.convolve(bugImg,receptiveFieldMatrix(lambda x,y: evenGabor2D(x,y,5,1)), mode='same')

imgplot = plt.imshow(Img_bugEvenGabor, cmap=cm.Greys_r)

# <markdowncell>

# Quadrature Pairs
# ------------------
#
# Now let's combine both gabor filters to see what will happen.

# <codecell>

def edgeEnergy(x,y,sigma, orientation):
    g1= oddGabor2D(x,y,sigma,orientation)
    g2= evenGabor2D(x,y,sigma,orientation)
    return(g1**2+g2**2)

# <codecell>

plotFilter(lambda x,y:edgeEnergy(x,y,50,0))

# <codecell>

Img_barEdgeEnergy = signal.convolve(barImg,receptiveFieldMatrix(lambda x,y: edgeEnergy(x,y,100,1)), mode='same')
imgplot = plt.imshow(Img_barEdgeEnergy, cmap=cm.Greys_r)

# <codecell>

Img_bugEdgeEnergy = signal.convolve(bugImg,receptiveFieldMatrix(lambda x,y: edgeEnergy(x,y,10,1)), mode='same')
imgplot = plt.imshow(Img_bugEdgeEnergy, cmap=cm.Greys_r)

# <codecell>
