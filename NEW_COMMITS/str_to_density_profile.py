'''
input: a density string like "a b c ..."
output: a density profile plot
'''
import matplotlib.pyplot as plt
import numpy as np
def plot_points(input_string):
    values = list(map(float, input_string.split()))
    x = list(range(1, len(values) + 1))
    
    # Plot the points
    plt.scatter(x, values, color='blue', label='$\langle  \hat{n}_i\\rangle$')
    plt.plot(x, values, linestyle='dashed', color='gray', alpha=0.7)  # Optional: line connecting points
    plt.axhline(np.sum(values)/len(values),alpha=0.5,label='$\\bar{\\nu}$')
    plt.xlabel("site")
    plt.ylabel("$\\nu$")
    plt.title("density profile (added sites = 3)")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig('/mnt/users/kotssvasiliou/tenpy/NEW_COMMITS/figures/MPS_figures/density_profile.png',dpi=500)

#######
str = '0.3333328 0.3333331 0.3333327 0.333333  0.3333339 0.3333342 0.333335  0.3333361 0.3333359 0.3333352 0.3333338 0.3333302 0.3333261 0.3333224 0.3333193 0.3333201 0.3333267 0.333339  0.333358  0.3333807 0.3333996 0.3334069 0.3333919 0.3333442 0.3332616 0.3331517 0.3330364 0.3329569 0.3329659 0.3331169 0.3334492 0.3339563 0.334561  0.3351048 0.3353485 0.3350069 0.3338287 0.3317022 0.32877   0.3255382 0.3229701 0.3225404 0.3260999 0.3352906 0.3503789 0.3687766 0.3842075 0.3877206 0.3706775 0.3303638 0.2761285 0.2274084 0.1913519 0.1525337 0.1160874 0.1363485 0.2364963 0.3673796 0.5551617 0.6824943 0.5459761 0.7332139 0.5956224'
plot_points(str)