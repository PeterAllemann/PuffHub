import numpy
from PIL import Image, ImageDraw

im = Image.open("270.jpg").convert("L")

imArray = numpy.asarray(im)

polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
mask = numpy.array(maskIm)

newImArray = imArray

newImArray = numpy.multiply(newImArray, mask)
nonZeroPoints = numpy.argwhere(newImArray)
topLeft = nonZeroPoints.min(axis=0)
bottomRight = nonZeroPoints.max(axis=0)
croppedArray = newImArray[topLeft[0]:bottomRight[0]+1, topLeft[1]:bottomRight[1]+1]

newIm = Image.fromarray(croppedArray, "L")
newIm.save("out.jpg")
