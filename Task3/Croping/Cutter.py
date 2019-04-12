import numpy
from PIL import Image, ImageDraw

# read image as RGB and add alpha (transparency)
im = Image.open("270.jpg").convert("RGBA")

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

'''
# argwhere will give you the coordinates of every non-zero point
true_points = np.argwhere(dat)
# take the smallest points and use them as the top left of your crop
top_left = true_points.min(axis=0)
# take the largest points and use them as the bottom right of your crop
bottom_right = true_points.max(axis=0)
out = dat[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
    top_left[1]:bottom_right[1]+1]  # inclusive
'''
