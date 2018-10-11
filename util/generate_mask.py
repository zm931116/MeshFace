from PIL import Image
import math
import operator
import matplotlib.pyplot as plt

image1 = Image.open('Figure_1.jpeg')
image1 = image1.rotate(10)
#image1.save('img1_10.jpg')

for i in range(1,4):
    img = Image.open('Figure_%d.jpeg' % i)
    for j in range(3):
        img = img.rotate((j+1)*15)
        img.save('./mask/mesh%d_%d.png' % (i, j))
