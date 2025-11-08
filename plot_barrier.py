# Plot barriers from Julia for 1d and 2d systems
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def Bar1(i,x, system):
    if (system == "simple"): # IBC
        # gamma = 0.5, alpha = 0.75 # manual gamma
        # if i == 0:
        #     return(-10.47513*x**3 + 77.52209*x**2 - 187.85372*x + 149.919)
        # else:
        #     return(0.63909*x**3 - 0.86426*x**2 + 0.37485*x + 0.00133)
        # gamma = 0.4916, alpha = 0.75 # gamma minimized
        if i == 0:
            return(-10.85069*x**3 + 79.98523*x**2 - 193.22042*x + 153.79698)
        else:
            return(0.63225*x**3 - 0.84767*x**2 + 0.3627*x)
    else: # logistic map kIBC v1 - TODO
        # gamma = 0.01, alpha = 0.3 # manual gamma
        if i == 0:
            return(193.77232*x**5+541.60798*x**4-3515.71744*x**3+5445.90085*x**2-3474.43221*x+809.0011)
        else:
            return(2.1214*x**5+37.98675*x**4-28.37654*x**3+7.07388*x**2-0.73407*x+0.03721)
        # gamma = , alpha = 0.3 # gamma minimized
        # if i == 0:
        #     return(0.0005*x**4 - 186.03829*x**3 + 514.21841*x**2 - 473.01827*x + 144.96383)
        # else:
        #     return(-0.00176*x**4 + 15.95292*x**3 - 8.00483*x**2 + 0.52444*x + 0.13245)


def plot1d(gamma,system):
    ell = 1
    if(system == "simple"):
        Xo = [2, 2.3] # initial set
        Xu = [1.6,1.9] # unsafe set
        X = [0,3] # state set
        ymin = -1
        ymax = 10
    else:
        Xo = [0.85, 0.95] # initial set
        Xu = [0.6, 0.7] # unsafe set
        X = [0,1] # state set
        ymin = 0
        ymax = 1.2
    B_bound = [gamma,1] # safety region upper bounds (B0 <= gamma, B1 < 1)

    colors = ['blue','m','g','r','b','k',"pink","olive","orange","purple"]
    state = np.linspace(X[0],X[1],150)
    B = np.zeros((ell+1,len(state)))
    for i in range(ell+1):
        B[i,:] = Bar1(i,state,system)
        plt.plot(state, B[i,:], label="B"+str(i), color=colors[i]) # plot barrier certificate
        plt.ylim(top=ymax,bottom=ymin)
        # plot nonnegative sublevel set of barriers (B_0(x) <= gamma or B_i(x)<1). y1, y2 decided based on plot without them
        plt.fill_between(state, y1 = ymin, y2 = ymax, where= np.logical_and(B[i,:]>= 0, B[i,:]<= B_bound[i]), facecolor=colors[-i-1], alpha=.2)
            
    # plot initial set region (only x axis)
    plt.fill_between(state,  -0.3, np.max(B), where= np.logical_and(Xo[0] <= state, state <= Xo[1]), facecolor='green', alpha=.3,label="Initial")
    # plot unsafe set region (only x axis)
    plt.fill_between(state,  -0.3, np.max(B), where= np.logical_and(Xu[0] <= state, state <= Xu[1]), facecolor='red', alpha=.3,label="Unsafe")
    plt.axhline(0,color='k') # horizontal line at B = 0
    plt.axvline(0,color='k') # vertical line at B = 0
    plt.axhline(1,color='r',linestyle='--') # horizontal line at B = 1
    plt.axhline(gamma,color='g',linestyle='--') # horizontal line at B = gamma
    plt.grid(True) 
    plt.legend(loc="upper left") #"best"
    plt.xlabel("x(t)")
    # save as svg, convert to eps online (transparency lost otherwise)
    plt.savefig(f"./media/ibc_l{ell}_{system}.svg", bbox_inches='tight', format='svg', dpi=1200)
    plt.show()

# IBC
def Bar2(j,x): 
    # j - index for B_j, x - states
    v, p = x[0], x[1]
    # gamma = 0.1, alphas = 0.44 # manual gamma
    if j == 0: # B_0
        return(0.00241*v**5+0.01149*v**4*p-0.057*v**3*p**2+0.09886*v**2*p**3-0.06707*v*p**4+0.03525*p**5-0.08654*v**4-0.11598*v**3*p+0.59542*v**2*p**2-1.06061*v*p**3+0.50893*p**4+1.01946*v**3+0.07164*v**2*p+0.36678*v*p**2-0.28549*p**3-4.41245*v**2-3.1385*v*p+2.24213*p**2+6.03*v+2.93683*p+10.55794)
    else:
        return(0.09978*v**5-0.02555*v**4*p+0.08103*v**3*p**2+0.00266*v**2*p**3+0.02174*v*p**4+0.01375*p**5-0.43666*v**4-0.22901*v**3*p-0.21965*v**2*p**2-0.18821*v*p**3-0.12012*p**4+0.71138*v**3+0.86508*v**2*p+0.57564*v*p**2+0.46029*p**3-0.40154*v**2-1.09595*v*p-0.78902*p**2-0.08266*v+0.78285*p+0.1533)
    # gamma = , alphas = 0.44 # gamma minimized
    # if j == 0: # B_0
    #     return()
    # else:
    #     return()

def plot2d(gamma):
    ell = 1
    # initial, unsafe sets and state space
    # First 2 for v, last 2 for p
    Xo = [6,7,2,3] # initial set (v,p).
    Xu = [3,5,0,3] # unsafe set
    X = [0,10,0,5] # state space

    n = 100 # number of sampled points
    v,p = np.linspace(X[0],X[1],n+1), np.linspace(X[2],X[3],n+1)
    x,y = np.meshgrid(v,p)
    fig, ax = plt.subplots()
    # filled contour for 0-sublevel set of barriers (B(x) >= 0 => -B(x) <= 0, B(x) <= gamma, B(x)<1)
    ax.contourf(v, p, Bar2(0,[x,y]), levels=[0,gamma],colors=['blue','w'],alpha=0.5)
    c = ax.contour(v, p, Bar2(0,[x,y]), levels=[0,gamma], colors=('m',), linewidths=(1,), origin='lower') # contours
    label1 = ax.clabel(c, fmt='%2.1f', colors='k', fontsize=11, manual = [(6.34,3.26)]) # label on contour
    ax.add_patch(Rectangle((6,2), 0, 0, color = 'blue',alpha=0.45,label="B0")) # added for legend, no rectangle plotted

    ax.contourf(v, p, Bar2(1,[x,y]),levels=[0,1],colors=['orange','w'],alpha=0.5)
    c = ax.contour(v, p, Bar2(1,[x,y]), levels=[0,1], colors=('m',), linewidths=(1,), origin='lower')
    label2 = ax.clabel(c, fmt='%2.1f', colors='k', fontsize=11, manual = [(1.25,3.56)])
    ax.add_patch(Rectangle((0,0), 0, 0,color = 'orange',alpha=0.5,label="B1")) # added for legend, no rectangle plotted
    for l in label1+label2: # how to add the labels on the plot
        l.set_weight("bold")
        l.set_rotation(0)
    
    # plot initial and unsafe regions
    ax.add_patch(Rectangle((Xo[0], Xo[2]), Xo[1]-Xo[0], Xo[3]-Xo[2],color = '#90ee90',alpha=1,label="Initial"))
    ax.add_patch(Rectangle((Xu[0], Xu[2]), Xu[1]-Xu[0], Xu[3]-Xu[2],color = 'red',alpha=0.9,label="Unsafe"))
    ax.axis('scaled')
    ax.grid()
    ax.legend(loc="lower right")
    plt.xlabel("$v$")
    plt.ylabel("$p$")
    plt.savefig(f"./media/ibc_2d_l{ell}.svg", bbox_inches='tight', format='svg', dpi=1200)
    plt.show()

gamma = [0.49, 0.01, 0.1] # get these from note above or txt file for optimal results if available
# plot1d(gamma[0],"simple")
# plot1d(gamma[1],"logistic")
# plot2d(gamma[2])
