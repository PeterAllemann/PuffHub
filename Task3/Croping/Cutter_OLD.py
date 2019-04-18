import numpy
from PIL import Image, ImageDraw

# read image as RGB and add alpha (transparency)
im = Image.open("270.jpg").convert("RGBA")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

# create mask
polygon = [(1092.00, 228.00), (1112.00, 227.00), (1432.00, 227.00), (1452.00, 226.00), (1575.00, 226.00), (1573.50, 147.00), (1412.00, 147.00), (1392.00, 146.00), (1152.00, 146.00), (1132.64, 141.16), (1131.45, 141.14), (1112.00, 146.00), (1034.00, 146.00), (1002.00, 228.00)]
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
mask = numpy.array(maskIm)

# assemble new image (uint8: 0-255)
newImArray = numpy.empty(imArray.shape,dtype='uint8')

# colors (three first columns, RGB)
newImArray[:,:,:3] = imArray[:,:,:3]

# transparency (4th column)
newImArray[:,:,3] = mask*255
for x in range(newImArray.shape[0]):
    for y in range (newImArray.shape[1]):
        if 0 == newImArray[x][y][3]:
            newImArray[x][y][0] = 0
            newImArray[x][y][1] = 0
            newImArray[x][y][2] = 0

# back to Image from numpy
newIm = Image.fromarray(newImArray, "RGBA")
newIm = newIm.crop(newIm.getbbox())
newIm.save("out.png")
