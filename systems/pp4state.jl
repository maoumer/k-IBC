# vector field - system variables
@polyvar x1 x2 x3 x4 w # w = noise
vars = [x1,x2,x3,x4]

# constants
r = 1.6; b1 = 0.3; d1 = 0.08; h1 = 20; 
alpha = 0.2; b2 = 0.3; d2 = 0.06; h2 = 20;
beta = 0.2; n = 0.3; d3 = 0.5; eta = 0.08;
d4 = 0.5;

T = 0.01; # sampling time
# noise constants
G0 = 0.05;
G1 = 0.01;
G2 = 0.005;
G3 = 0;

# system dynamics - discrete-time 4 state prey predator type model
function dyn(x,w)
    x1,x2,x3,x4 = x
     f = [x1 + T*(r*x2 - (b1+d1)*x1 -h1*x1*x4) + G0*w,
          x2 + T*(b1*x1 - b2*x2*x2 - h2*x2*x4 - d2*x2) + G1*w,
          x3 + T*(alpha*h1*x1*x4 + beta*h2*x2*x4 - (n+d3)*x3 -eta*x2*x4),# + G2*w,
          x4 + T*(n*x3 - d4*x4),# + G3*w
        ]
     return f
end

#initial, unsafe sets and state space
Xo = [6.5,7, 5.5,6, 4.5,5, 3.5,4] # initial set (v,p)
Xu = [3,5, 3,4, 0,8, 0,5] # unsafe set
X = [0,10, 0,10, 0,10, 0,10] # state space

# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
go = [x1-Xo[1], Xo[2]-x1, 
      x2-Xo[3], Xo[4]-x2,
      x3-Xo[5], Xo[6]-x3, 
      x4-Xo[7], Xo[8]-x4] 
gu = [x1-Xu[1], Xu[2]-x1, 
      x2-Xu[3], Xu[4]-x2,
      x3-Xu[5], Xu[6]-x3, 
      x4-Xu[7], Xu[8]-x4] 
g = [x1-X[1], X[2]-x1, 
     x2-X[3], X[4]-x2,
     x3-X[5], X[6]-x3, 
     x4-X[7], X[8]-x4]
