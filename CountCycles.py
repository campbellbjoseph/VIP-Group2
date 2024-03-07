import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'xx-small',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
plt.rcParams.update(params)
import numpy as np
import math
from math import factorial

HStress, HStrain, VStress=np.loadtxt('5_3 (3).csv',delimiter=',',skiprows=1, unpack=True)


#Smoothing function defined here
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


#Smoothing the experimental data here
HStress_smoothed=savitzky_golay(HStress,31,5)
HStrain_smoothed=savitzky_golay(HStrain,31,5)
VStress_smoothed=savitzky_golay(VStress,31,5)

#Finding Sign change points - change from postive shear stress to negative shear stress
a=HStress_smoothed
asign = np.sign(a)
signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
index=np.where(signchange==1)[0]  #This has list of all indices where the sign change happens
cycles = index[1::2]   #This has list of indices where the sign changes from negative to positive.
cycles_true=np.zeros(len(cycles))

#Finding true indices of start of new cycles
for i in range(0,len(cycles)):
    if (abs(HStress_smoothed[cycles[i]-1]) < HStress_smoothed[cycles[i]]):
        cycles_true[i]=cycles[i]-1
    else:
        cycles_true[i]=cycles[i]

cycles_true=np.int64(cycles_true)

print('Number of cycles=', len(cycles_true))

#Plotting the important figures
# plt.figure(1)
# plt.plot(HStrain,HStress) #orginal test data
# plt.plot(HStrain_smoothed,HStress_smoothed) #smoothed test data
# for k in cycles_true:
#     plt.plot(HStrain_smoothed[k],HStress_smoothed[k],'o',color='red',alpha=0.5) #Highlighting starting points of the cycles
# plt.xlabel('Shear Strain (%)')
# plt.ylabel('Shear Stress (kPa)')

# plt.figure(2)
# plt.plot(VStress,HStress)  #orginal test data
# plt.plot(VStress_smoothed,HStress_smoothed) #smoothed test data
# for k in cycles_true:
#     plt.plot(VStress_smoothed[k],HStress_smoothed[k],'o',color='red',alpha=0.5) #Highlighting starting points of the cycles
# plt.xlabel('Vertical Stress (kPa)')
# plt.ylabel('Shear Stress (kPa)')


######### GROUP 2 WORK: #########
## If you are modifiying this file, the relevant values to change are in lines 127-130. Depending on the number of cycles,
## calculated via len(cycles_true), it may be possible to develop nicer diagrams. I am not sure if they want us to have them be 3 wide, 
## but look on your own to see what works best. 

width = 36
height = 36
rows = 5
cols = 1

plt.figure(figsize=(width, height)) # figsize is (width, height) in inches 
for i in range(len(cycles_true) - 1):
    plt.subplot(rows,cols,i+1) # subplot(x, y, i+1) means the overall figure has x rows, y cols, and that this is the i+1th subplot
    plt.plot(HStrain_smoothed[cycles_true[i]:cycles_true[i+1]], HStress_smoothed[cycles_true[i]:cycles_true[i+1]])
    plt.title("Cycle " + str(i))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=.27, hspace=1.2)  # !!!!Adjust values wspace and hspace as needed to make pretty
 

# Plotting the last cycle, a bit annoying
plt.subplot(rows,cols,len(cycles_true))
plt.plot(HStrain_smoothed[cycles_true[-1]:], HStress_smoothed[cycles_true[-1]:])
plt.title("Cycle " + str(len(cycles_true) - 1))

plt.show()

