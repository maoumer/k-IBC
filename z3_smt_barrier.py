# Takes way too long to compute
# Computes IBC for 1d stochastic system using SMT solver
from z3 import * # SMT solver
import numpy as np
from matplotlib import pyplot as plt
import time

dim = 1 # system state dimension
gamma = 0.5 # upper bound of B_0
w = Real("w") # noise term
def f(x,w):
    return(x/2 + 0.05*w)

# monomial for cubic barrier certificate template
def ret_monoms(x):
    return(1, x, x**2, x**3)

leng = len(ret_monoms(0)) # number of monomial terms

# barrier certificate template - cubic
def Bar(i,x,coeffs_smt): #B_i, x - states
    return(np.dot(ret_monoms(x),coeffs_smt[i]))

#z3 real to python float
def realToFloat(var):
    var = var.as_fraction()
    return(float(var.numerator) / float(var.denominator))

# hard coded for normal distribution, n>=1
def moments(n):
    mu, sigma = 0, 1
    if (n%2 == 1):
        return(0)
    elif(n==2):
        return(1*sigma**n)
    elif(n==4):
        return(3*sigma**n)
    elif(n==6):
        return(15*sigma**n)
    elif(n==8):
        return(105*sigma**n)


# expectation of a function of x and w. Replace w^j terms with moments Exp[w^j]
def Exp(xw): 
    # https://ericpony.github.io/z3py-tutorial/guide-examples.htm
    # print(help_simplify())
    monomial_sums = simplify(xw, som=True) # simplify as monomial sums of each term
    sum = monomial_sums.arg(0) # first term. Usually constant

    # go through 2nd up to last term
    for i in range(1,monomial_sums.num_args()):
        # collapse w*...*w etc to w^k in each of the terms
        comp=simplify((monomial_sums.arg(i)).arg(1), mul_to_power=True)

        #start from largest degree of monomials
        # for j in range(leng-1,1,-1):
        # hard coded for leng = 4 (max degree of monomial = 3). TODO: Find a way to compute for arbitrary length
        if(w**3 in (comp).children()):
            subbed = substitute(monomial_sums.arg(i), (w,RealVal(1))) #replace all w's in term with 1 as they are written w*...*w
            subbed *= RealVal(moments(3)) #  multiply replaced term with Exp(w^j)
        elif(w**2 in (comp).children()):
            subbed = substitute(monomial_sums.arg(i), (w,RealVal(1))) # replace all w's in term with 1 as they are written w*...*w
            subbed *= RealVal(moments(2)) #  multiply replaced term with Exp(w^j)
        elif(w in (comp).children()):
            subbed = substitute(monomial_sums.arg(i), (w, RealVal(moments(1))))
        else:
            subbed = monomial_sums.arg(i)
        sum += subbed
    return(simplify(sum, mul_to_power=True))


# IBC computation. kmax = 0 -> standard BC
def IBC(kmax, Xo, Xu, X, length = leng):
    # kmax - max k value for B_k in IBC, Xo - initial, Xu - unsafe, X - state set, length - hyperparameter pass
    coeffs_smt = [] # store coefficients for all B_i
    for i in range(kmax+1): # for each B_i upto max k
        t = [] # store coefficients for each B_i
        for j in range(length): # for each term in monomial
            c = Real("coeff"+str(i)+","+str(j)) # construct a coefficient
            t.append(c)
        coeffs_smt.append(t)
    
    xvar = [] # state variables
    for i in range(dim):
        xvar.append(Real("x"+str(i)))

    # nonnegative condition
    nonneg_cond = And(True)
    for i in range(kmax+1):
        nonneg_cond = And(nonneg_cond, Bar(i,xvar[0],coeffs_smt) >= 0) # [0] for only 1d system
    nonneg_cond = ForAll(xvar, nonneg_cond)

    # initial state
    initial_cond = ForAll(xvar, 
                    Implies(And(Xo[0] <= xvar[0], xvar[0] <= Xo[1]), # if initial state
                            Bar(0,xvar[0],coeffs_smt) <= gamma)) # B_0 <= gamma condition

    # unsafe state
    quer = And(True)
    for i in range(kmax+1): # for each B_i
        quer = And(quer, Bar(i,xvar[0],coeffs_smt) >= 1) # condition on all B_i
    unsafe_cond = ForAll(xvar, 
                         Implies(And(Xu[0] <= xvar[0], xvar[0] <= Xu[1]), #if unsafe
                                quer)) # then check conditions on all B_i

    # 4th condition (if kmax > 0)
    quer = And(True)
    for j in range(kmax): # 4th condition E[B_(i+1)(f(x,w))|x] <= alpha_i B_i(x), alpha_i = 1
        quer = And(quer, Exp(Bar(j+1, f(xvar[0],w), coeffs_smt)) <= Bar(j, xvar[0], coeffs_smt))
    quer = ForAll(xvar,
                    Implies(And(X[0] <= xvar[0], xvar[0] <= X[1]),
                        quer))

    # 5th condition E[B_k(f)] => B_k() (invariance)
    last_cond = ForAll(xvar, 
                    Implies(And(X[0] <= xvar[0], xvar[0] <= X[1]),
                        Exp(Bar(kmax, f(xvar[0],w), coeffs_smt)) <= Bar(kmax, xvar[0], coeffs_smt)) )
    last_cond = And(quer, last_cond) # 4th + 5th cond
    Bar_cond = And(nonneg_cond, initial_cond, unsafe_cond, last_cond)
    
    # compile everything and search ibc
    s = Solver()
    print(f"k = {kmax}:")
    s.add(Bar_cond)
    start_time = time.time()
    status = s.check() # z3 expecting to run s.check() only once so save it in variable
    print(f"Status: {status}")
    print(f"Execution time: {time.time() - start_time}") # ~ s
    if (status == sat):
        return(s.model())
    elif (status == unsat):
        # c = s.unsat_core()
        return(None)

k_max = 1 # max bound
Xo = [2, 2.3] # initial set
Xu = [1.6,1.9] # unsafe set
X = [0,3] # state set
colors = ['blue','m','g','r','b','k',"pink","olive","orange","purple"]
for k in range(k_max,0-1,-1): # goes from k_max to 0
    model = IBC(k,Xo,Xu,X)
    if model != None: # if model successfully computed
        state = np.linspace(X[0],X[1],100)
        B = np.zeros((k+1,len(state))) # need +1 because of 0 indexing
        # sort model coefficients alphabetically by name (ascendingly for each i in B_i)
        sorted_model = sorted([(var, model[var]) for var in model], key = lambda x: str(x[0]))
        start = 0 # starting index for each B_i
        for i in range(k+1): # for i from 0 upto and including k
            # collect relevant coefficients for each B_i (start to (not including) start+leng gives the right number of monomial terms)
            # and reverse it to get highest to lowest degree ([::-1])
            coeffs = np.array([realToFloat(val) for (_, val) in sorted_model[start:start+leng]])[::-1]
            start += leng # shift starting index
            # coeffs needs to be highest to lowest degree i.e. coeffs[0]*x + coeffs[1]*1.
            print(f"i={i}: {coeffs}")
            B[i,:] = np.polyval(coeffs, state) # evaluate barrier certificate
            plt.plot(state, B[i,:], label="B"+str(i), color=colors[i]) # plot barrier certificate
            # plot 0-sublevel set of barriers (B(x) <= 0). y1, y2 decided based on plot without them
            plt.fill_between(state, y1 = -1.5625, y2 = 2, where= (B[i,:]<=0), facecolor=colors[-i-1], alpha=.2)
        
        # plot initial set region (only x axis)
        plt.fill_between(state, np.min(B), np.max(B), where= np.logical_and(Xo[0] <= state, state <= Xo[1]), facecolor='green', alpha=.3,label="Initial")
        # plot unsafe set region (only x axis)
        plt.fill_between(state, np.min(B), np.max(B), where= np.logical_and(Xu[0] <= state, state <= Xu[1]), facecolor='red', alpha=.3,label="Unsafe")
        plt.axhline(0,color='k') # horizontal line at B = 0
        plt.grid(True) 
        plt.legend(loc="upper left") #"best"
        plt.xlabel("x(t)")
        plt.savefig(f"./media/ibc_stoch_k{k}.svg", bbox_inches='tight', format='svg', dpi=1200)
        plt.show()
        break # once IBC found, no need to check for other k

