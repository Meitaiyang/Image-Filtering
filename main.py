import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def mean_filter(img):
    
    mean_size = 3
    ker = np.array([[1]*mean_size]*mean_size)/(mean_size**2)
    xs = img.shape[0]-(ker.shape[0]-1)
    ys = img.shape[1]-(ker.shape[1]-1)
    conv_mat = np.zeros((xs,ys))
    
    for i in range(xs):
        for j in range(ys):
            conv_mat[i][j] = (img[i:i+ker.shape[0],j:j+ker.shape[1]]*ker).sum()
    
    return conv_mat.astype(np.uint8)

def median_filter(img):
    median_size = 3
    ker = np.array([[1]*median_size]*median_size)
    xs = img.shape[0]-(ker.shape[0]-1)
    ys = img.shape[1]-(ker.shape[1]-1)
    conv_mat = np.zeros((xs,ys))
    
    for i in range(xs):
        for j in range(ys):
            conv_mat[i][j] = np.median(img[i:i+ker.shape[0],j:j+ker.shape[1]])

    return conv_mat.astype(np.uint8)

def img_hist(name,img):


    xs = img.shape[0]
    ys = img.shape[1]
    x = range(0,256)
    y = [0]*256
    for i in range(xs):
        for j in range(ys):
            y[img[i][j]] += 1

    fig = plt.figure(name)
    plt.ylim(0,4000)
    plt.bar(x,y)
    plt.savefig(name)

if __name__ == '__main__':
    
    image = cv2.imread('noise_image.png',0) #read the image
    result_folder = "result"
    
    pathlib.Path(f"{result_folder}").mkdir(parents=True, exist_ok=True)
    
    image_mean = mean_filter(image)
    image_median = median_filter(image)
    img_hist(result_folder+'/img_hist',image)
    img_hist(result_folder+'/img_mean_hist',image_mean)
    img_hist(result_folder+'/img_median_hist',image_median)
    cv2.imwrite(result_folder+'/img_mean.png',image_mean)
    cv2.imwrite(result_folder+'/img_median.png',image_median)
    cv2.waitKey(0)