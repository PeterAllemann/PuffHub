import numpy
from PIL import Image, ImageDraw
import svgpathtools as svg

# Read SVG file
paths = [0]*15
attributes = [0]*15

# Get trainng paths
for i in range(10):
    paths[i], attributes[i] = svg.svg2paths('../ground-truth/locations/27' + str(i) + '.svg')

# Get test paths
for i in range(10,15):
    paths[i], attributes[i] = svg.svg2paths('../ground-truth/locations/30' + str(i-10) + '.svg')

# convert for conveniance or how ever this should be spelled
polygons = []
ids = []
for h in range(len(paths)): # File 270 to 304
#    for h in range(len(paths[i])) # number of words per image
    for i in range(len(paths[h])): # number of words per image
        polygon = []
        for j in range(len(paths[h][i])): # length of each path/polygon
            polygon.append((paths[h][i][j][0].real, paths[h][i][j][0].imag))
        polygons.append(polygon)
        ids.append(attributes[h][i]['id'])

for i in range(len(polygons)):
    # read image as RGB and add alpha (transparency)
    im = Image.open('../images/' + ids[i][0:3] + '.jpg' ).convert("RGBA")

    # convert to numpy (for convenience)
    imArray = numpy.asarray(im)

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygons[i], outline=1, fill=1)
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    nonZeroCoords = numpy.argwhere(newImArray[:,:,3])
    topLeft = nonZeroCoords.min(axis=0)
    bottomRight = nonZeroCoords.max(axis=0)
    croppedImArray = newImArray[topLeft[0]:bottomRight[0]+1, topLeft[1]:bottomRight[1]+1]

    # back to Image from numpy
    newIm = Image.fromarray(croppedImArray, "RGBA")
    newIm = newIm.crop(newIm.getbbox())
    newIm.save('../Cropped-images/' + ids[i] + '.png')
