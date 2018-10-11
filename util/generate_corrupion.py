import matplotlib.pyplot as plt
import math

import numpy as np

plt.figure(1)

x=[]
y=[]
y1 = []
# mag = np.random.randint(8,12)
# w = np.random.randint(4,6)
mag = 12
w = 7
c = 2.9
angle = 45

a=np.linspace(0,10,10000)

print(mag)
#print(w)
for j in range(5):
    x = []
    y = []
    y1 = []
    for i in a:
        x.append(i)
        y.append(mag * math.sin(w * i) + mag * (c*j ))
        #xr = x * plt.cosd(45)- y * plt.sind(45)
        #yr = x* plt.sind(angle) + y* plt.sind()
        y1.append(mag * math.sin(-w * i) + mag * (c*j +c /2 ))


    plt.plot(x,y,color = 'black')
    plt.plot(x,y1,color ='black')

plt.show()
