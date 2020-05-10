import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def make_edge_image(filename, percentile_threshold=80, show=False, height=100):
    
    #define the vertical filter
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]

    #define the horizontal filter
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

    #read in the pinwheel image
    img = plt.imread(filename)
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)

    width = height*img.shape[1]/img.shape[0]
    img = np.array(Image.fromarray(img).resize((100,100)))

    #get the dimensions of the image
    n,m = img.shape

    #initialize the edges image
    edges_img = img.copy()

    #loop over all pixels in the image
    for row in range(3, n-2):
        for col in range(3, m-2):
            
            #create little local 3x3 box
            local_pixels = img[row-1:row+2, col-1:col+2]
            
            #apply the vertical filter
            vertical_transformed_pixels = vertical_filter*local_pixels
            #remap the vertical score
            vertical_score = vertical_transformed_pixels.sum()/4
            
            #apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            #remap the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum()/4
            
            #combine the horizontal and vertical scores into a total edge score
            edge_score = (vertical_score**2 + horizontal_score**2)**.5
            
            #insert this edge score into the edges image
            edges_img[row, col] = edge_score*3

    #remap the values in the 0-1 range in case they went out of bounds
    edges_img = edges_img/edges_img.max()

    boundary = np.percentile(edges_img, percentile_threshold)

    edges_img = np.where(edges_img > boundary, 1, 0)
    edges_img = edges_img.astype(int)
    edges_img = np.rot90(edges_img)

    np.savetxt(f"../data/{filename}.dat", edges_img, fmt="%d")

    if show:
        plt.imshow(edges_img)
        plt.show()
        

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        height = int(sys.argv[2])
    else:
        height = 100
    if len(sys.argv) > 3:
        show = bool(sys.argv[3])
    else:
        show = False
    if len(sys.argv) > 4:
        percentile_threshold = float(sys.argv[4])
    else:
        percentile_threshold = 80
    
    
    make_edge_image(filename, percentile_threshold, show, height)