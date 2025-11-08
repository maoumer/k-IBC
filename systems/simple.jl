# vector field
@polyvar x # z # x for state, w for noise, z for replacing higher powers of w
vars = [x]

# constants
G = 0.05

# dynamics
function dyn(x,w)
    x = x[1]
    f = [x/2 + G*w]
    return f
end

# f = dyn(vars,w) #dyn_n(dyn,1)(vars) # f(x)

#initial, unsafe sets and state space
Xo = [2, 2.3] # initial set
Xu = [1.6,1.9] # unsafe set
X = [0,3] # state space

# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
go = [x-Xo[1], Xo[2]-x] #initial
gu = [x-Xu[1], Xu[2]-x] #unsafe
g = [x-X[1], X[2]-x] #whole
