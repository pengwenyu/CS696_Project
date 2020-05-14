import matplotlib.pyplot as plt # plt
import matplotlib.image as mpimg # mpimg
from skimage import io,transform

img = io.imread('./wrongnoGlass/010760.jpg')

img = transform.resize(img, (200, 200))

plt.imshow(img)
plt.axis('off')
plt.show()
