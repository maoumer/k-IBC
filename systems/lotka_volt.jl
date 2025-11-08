# vector field - system variables
@polyvar v p
vars = [v,p]

# constants
alpha = 1.1; #growth rate -> prey
beta  = 0.4; #death rate -> prey
gamm = 0.4; #death rate -> predator
delta = 0.1; #growth rate -> predator
T = 0.1; # sampling time
# noise constants
G0 = 0.01; #0.05
G1 = 0.005; #0.01

# system dynamics - discrete-time Lotka-Volterra type model
function dyn(x,w)
     v, p = x
     f = [v + T*(alpha*v*(1-v) - beta*v*p) + G0*w,
          p + T*(gamm*p*(p*0-1) + delta*v*p) + G1*w]
     return f
end

#initial, unsafe sets and state space
Xo = [6,7,2,3] # initial set (v,p)
Xu = [3,5,0,3] # unsafe set
X = [0,10,0,5] # state space

# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
go = [v-Xo[1], Xo[2]-v, p-Xo[3], Xo[4]-p] #initial
gu = [v-Xu[1], Xu[2]-v, p-Xu[3], Xu[4]-p] #unsafe
g  = [v-X[1],  X[2]-v,  p-X[3],  X[4]-p] #whole
