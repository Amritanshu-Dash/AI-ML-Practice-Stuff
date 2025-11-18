import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Animation function
def animate(frame): # this function is called 90 times for frames from 0 to 89 ass specified in the FuncAnimation
  theta = np.radians(frame) * 4 # converts frame in radian to theta, multiplied 4 to make it 360 .....
  T = np.array([  [np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]
                ])
  ## we are here creating a transformation matrix purely math based
  rotated_vector = T @ vector ##matrix multiplication
  arrow.set_data([0, rotated_vector[0]], [0, rotated_vector[1]]) ##the list1 here specifying the starting part origin and end for arrow and list2 ending of arrow for that specific angle.
  ax.set_title(f"Rotation: {frame}°")
  return arrow, #as arrow is a list so we are using ',' to unpack it because our goal is to just find out the changed value thats it so it return the starting values

#Initial vector
vector = np.array([4, 2])

#plot setup
fig, ax = plt.subplots(figsize=(6, 6)) ## return window and the pointer of inside we can name them way we want fig->window , ax->the plotting inside
ax.set_xlim(-5, 5) ## sets the x limits of the plot inside the window that will be shown on x axis
ax.set_ylim(-5, 5) ## sets the y limits of the plot inside the window that will be shown on y axis
ax.grid(True)
ax.axhline(0, color='black', linewidth=0.5) ##the origin and color and width of the x-axis or horizontal lines
ax.axvline(0, color='black', linewidth=0.5) ##the origin and color and width of the y-axis or vertical lines
arrow, = ax.plot([], [], lw=3, color='blue') ##initially empty plot for arrow which will be updated in animate function

#Create animation
ani = FuncAnimation(fig, animate, frames=90, interval=50, blit=True) ##frames is how many frames we want in our animation kinda recursion, interval is time between each frame in milliseconds, blit true means only the parts that have changed will be redrawn making it more efficient
ani.save('rotation_animation.gif', writer='pillow') ##saving the animation as gif using pillow library
plt.show()
