import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

import imageio

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import lagrange

from PIL import Image

import statistics
import numpy as np

import math

# Returns the value corresponding to x for a Normal
# distribution with mean mu and standard deviation sig
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

sq = np.array(Image.open("./cluster_tiff.tif"))
plt.imshow(sq, cmap='gray')
plt.show()

matplotlib.image.imsave('inverse_squared.png', sq, cmap='gray')

# Sample of background noise from the same MRI scan
# the array may be any size but larger samples of noise 
# yield better results
background_sample = np.load('Background_Sample.npy')

print('Attempting R2 Quantification...')
counts, bins, _ = plt.hist(background_sample, bins=100, density=1)

# Display the result of Gaussian/Normal fitting the background noise
muo, sigma = norm.fit(background_sample)
best_fit_line = norm.pdf(bins,muo,sigma)
plt.plot(bins, best_fit_line)
plt.show()

# Fade colors between different mu values to ensure
# distinguishabilty
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='red'
c2='blue'

S1_arr = []
S2_arr = []
S3_arr = []
S4_arr = []
S5_arr = []

sweep_range = 5

for mu in range(int(muo)-sweep_range,int(muo)+sweep_range):
    v10x_S1_diff = []
    v10x_S2_diff = []
    v10x_S3_diff = []
    v10x_S4_diff = []
    
    v10x_S1_Maxes = sq
    maxgauss = gaussian(mu,mu,sigma)
    v10x_S1_Maxes[v10x_S1_Maxes > mu] = mu

    res_S1 = (maxgauss-gaussian(v10x_S1_Maxes,mu,sigma))/maxgauss

    for i in range(len(v10x_S1_Maxes)):
        v10x_S1_diff.append(v10x_S1_Maxes[i]*res_S1[i])

    abcd = np.array(gaussian_filter(v10x_S1_diff,1))# was 2
    asdf = 1/np.log(abcd)
    
    # Diffusion coefficient in m^2/s
    D = 2.5e-9*3
    
    # Use these indices to select the regions of the image
    # for sampling of each condition
    C3 = [asdf[18:20,10:12]]
    C2 = [asdf[18:20,7:9]]
    C1 = [asdf[18:20,5:7]]
    C3 = np.array(C3).flatten()
    C2 = np.array(C2).flatten()
    C1 = np.array(C1).flatten()

    # Diffusion time constant in milliseconds
    TD = np.multiply([((25e-9)**2/D),((50e-9)**2/D),((100e-9)**2/D)],1000)
    R2 = np.log([(np.mean(C1)),(np.mean(C2)),(np.mean(C3))])

    plt.plot(TD,R2,color=colorFader(c1,c2,(mu-(int(muo)-500))/1000))

    S1_arr.append(np.mean(C1))
    S2_arr.append(np.mean(C2))
    S3_arr.append(np.mean(C3))

print('TD:')
print(TD)
print('R2:')
print(R2)
plt.xscale('log')
plt.show()