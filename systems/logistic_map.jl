# vector field
@polyvar x # z # x for state, w for noise, z for replacing higher powers of w
vars = [x]

# constants
r = 1.5;
G = 0.0005;

# dynamics - logistic_map model
function dyn(x,w)
    x = x[1]
    f = [r*x*(1-x) + G*w]
    return f
end

#initial, unsafe sets and state space
Xo = [0.85, 0.95] # initial set
Xu = [0.6, 0.7] # unsafe set 
X = [0,1] # state space

# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
go = [x-Xo[1], Xo[2]-x] #initial
gu = [x-Xu[1], Xu[2]-x] #unsafe
g = [x-X[1], X[2]-x] #whole
