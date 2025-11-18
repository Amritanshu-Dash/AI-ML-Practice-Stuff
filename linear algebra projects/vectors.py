import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

add = v1 + v2
scale_v2 = 2 * v2

#plot
#plt.arrow(start x, start y, ending x (dx), ending y (dy), color, width, head width of arrow, legend or label used in legend)
plt.figure(figsize=(6, 6))
plt.arrow(0, 0, v1[0], v1[1], color='blue', width=0.1, head_width=0.3, label='v1')
plt.arrow(0, 0, v2[0], v2[1], color='red', width=0.2, head_width=0.6, label='v2')
plt.arrow(0, 0, add[0], add[1], color='green', width=0.1, head_width=0.3, label='add=v1+v2')
plt.arrow(0, 0, scale_v2[0], scale_v2[1], color='orange', width=0.1, head_width=0.3, label='scale of v2(2*v2)')

plt.xlim(0, 12); plt.ylim(0, 12) #zoom control what it does it make an 8 * 8 square mean takes x from 0 to 8 and y also 0 to 8 if we dont use it might not give us good square or plot
plt.grid(True)
plt.legend()
plt.title("Vector operations of adding and scaling")
plt.show()
