import numpy
from PIL import Image, ImageDraw

# read image as RGB and add alpha (transparency)
#im = Image.open("270.jpg").convert("RGBA")
im = Image.open("270.jpg").convert("L")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

# create mask
polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
mask = numpy.array(maskIm)

# assemble new image (uint8: 0-255)
newImArray = numpy.empty(imArray.shape,dtype='uint8')

# colors (three first columns, RGB)
newImArray = imArray

# transparency (4th column)
newImArray = numpy.multiply(newImArray, mask)

# back to Image from numpy
newIm = Image.fromarray(newImArray, "L")
newIm = newIm.crop(newIm.getbbox())
newIm.save("out.jpg")
