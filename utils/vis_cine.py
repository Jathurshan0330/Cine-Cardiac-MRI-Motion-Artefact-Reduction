import numpy as np
from ipywidgets import  interactive, IntSlider
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib.animation import FuncAnimation
from IPython import display



def interactive_gif_data(img, cmap = 'gray'):
  """
  Interactive tool to visualize the dataset as gif. 
  """
  def Animate(slice):
    Figure = plt.figure() 
    # creating a plot
    imgs_plotted = plt.imshow(img[slice,0,:,:], cmap = 'gray') 
    plt.axis("off")

    def AnimationFunction(frame):
        y = img[slice,frame,:,:]
    
        # line is set with new values of x and y
        imgs_plotted.set_data(y)

    anim_created = FuncAnimation(Figure, AnimationFunction, frames=img.shape[1]-1, interval=100)

    video = anim_created.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    # plt.show()
  # good practice to close the plt object.
    plt.close()

  select_slice = IntSlider(min=0, max=img.shape[0]-1, description='Select slice', continuous_update=True)


  return interactive(Animate, slice=select_slice)

def AnimationFunction(frame):
        y = img[slice,frame,:,:]
    
        # line is set with new values of x and y
        imgs_plotted.set_data(y)