import numpy
from PIL import Image, ImageDraw
import svgpathtools as svg

# read image as RGB and add alpha (transparency)
im = Image.open("270.jpg").convert("RGBA")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

# Read SVG file
paths = [0]*15
attributes = [0]*15

#pathx, attributex = svg.svg2paths('../ground-truth/locations/270.svg')
#print(len(pathx[0]))

'''
for i in range(1):
    paths[i], attributes[i] = svg.svg2paths('../ground-truth/locations/27' + str(i) + '.svg')
print(paths[0][0][0][0].imag)
'''

polygons = []

for i in range(len(paths)):
    for j in range(len(paths[i][0])):
        for k in range (len(paths[i][0][j])):
            polygons.append((paths[i][0][j][k].real, paths[i][0][j][k].imag))
print(polygons[0])

# Path length
# print(len(paths[0]))
# Path id
# print(attributes[0]['id'])
# Path x coord
# print(paths[0][0][0].real)
# Path y coord
# print(paths[0][0][0].imag)

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


'''
for x in range(newImArray.shape[0]):
    for y in range (newImArray.shape[1]):
        if 0 == newImArray[x][y][3]:
            newImArray[x][y][0] = 0
            newImArray[x][y][1] = 0
            newImArray[x][y][2] = 0
'''

# back to Image from numpy
newIm = Image.fromarray(newImArray, "RGBA")
newIm = newIm.crop(newIm.getbbox())
newIm.save("out.png")
