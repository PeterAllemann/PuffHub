"""
Cropps
"""

import numpy
from PIL import Image, ImageDraw
import svgpathtools as svg
from skimage import filters

# Get training paths
paths = [0] * 15
attributes = [0] * 15

for i in range(10):
    paths[i], attributes[i] = svg.svg2paths('../ground-truth/locations/27' + str(i) + '.svg')

# Get test paths
for i in range(10, 15):
    paths[i], attributes[i] = svg.svg2paths('../ground-truth/locations/30' + str(i - 10) + '.svg')

# convert for convenience
polygons = []
ids = []
for h in range(len(paths)):  # File 270 to 304
    for i in range(len(paths[h])):  # number of words per image
        polygon = []
        for j in range(len(paths[h][i])):  # length of each path/polygon
            polygon.append((paths[h][i][j][0].real, paths[h][i][j][0].imag))
        polygons.append(polygon)
        ids.append(attributes[h][i]['id'])

# Cut and save words as separate images
for i in range(len(polygons)):
    # read image as greyscale
    im = Image.open('../images/' + ids[i][0:3] + '.jpg').convert('L')

    # convert to array
    imArray = numpy.asarray(im)

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygons[i], outline=1, fill=1)
    mask = numpy.array(maskIm)

    # cut trailing edges
    nonZeroCoords = numpy.argwhere(mask)
    topLeft = nonZeroCoords.min(axis=0)
    bottomRight = nonZeroCoords.max(axis=0)
    mask = mask[topLeft[0]:bottomRight[0] + 1, topLeft[1]:bottomRight[1] + 1]
    imArray = imArray[topLeft[0]:bottomRight[0] + 1, topLeft[1]:bottomRight[1] + 1]

    # binarize
    newImArray = numpy.copy(imArray)
    newImArray[0 == mask] = 255
    threshold = filters.threshold_sauvola(newImArray)
    newImArray[newImArray <= threshold] = 0
    newImArray[newImArray > threshold] = 255

    # convert array to image
    newIm = Image.fromarray(newImArray, 'L')
    newIm.save('../Cropped-images/' + ids[i] + '.png')
