# Computes barrier certificate (standard vs IBC vs k-BC vs k-IBC v1) using SOS for stochastic systems

# include important libraries
using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using Distributions
using TaylorSeries
using SpecialFunctions
using TSSOS # important for SOS

sos_tol = 1 # the maximum degree of unknown SOS polynomials = deg + sos_tol 
error = 5   # precision digit places
α = [0.75, 0.44, 0.3, 0.3] #simple, lotka_volt, logistic map, pp4state
c = [1e-2, 1e-2, 5e-3, 5e-3] # for k-induction, 1e-2 for simple, lotka_volt, 5e-3 for logistic map, pp4state
threshold = 1e-4 # for difference between primal and dual objective values

# syntax to call f^n(x) = f^(n-1) ∘ f(x) ==> dyn_n(f, n)(x)
# f = dyn_n(dyn,1)(vars)
function dyn_n(f, n) # f (function) composed n (scalar) times, n ≥ 1 for sanity
    function (x,w)  # <- this is syntax for an anonymous function
        if n>0 
            val = f(x,w[1])
            for j in 2:n
                val = f(val,w[j]) # different noise terms at each step, discussed in paper
            end
        else
            val = x # f^0 = identity
        end
        return val
    end
end

# generate moments using n^th derivative of MGF (moment generating function)
# used to replace E[w^j] terms later 
function moments(n, distribution = "normal")
    # distribution = type of probability distribution
    # n = the derivative of MGF, ≥ 1

    # using Taylor Series -- working solution
    if distribution == "normal"
        μ = 0
        σ = 1
        MGF(t) = exp(t*μ + 0.5*σ^2*t^2) # mgf of normal distribution
    end

    s = Taylor1(n) # taylor series over this variable s centered at 0
    # MGF(s) gives the Taylor expansion of MGF up to order n
    nth_moment = getcoeff(MGF(s), n)*gamma(n+1) # n^th Taylor coefficient = f^{(n)}(0) / n!, gamma(n+1) = n! for n≥0
    # println(nth_moment)
    return nth_moment
end

# replace the noise term w in expression Bf with the moment
# expecting w as a vector
function replaceW(Bf,w)
    E_Bf = 0 # expectation of B(f(x,w))  or B(f^k(x,w))
    for term in Bf # go through the terms of B(f(x,w))
        for wi in w # go through the noise terms in w for each step
            w_deg = degree(term, wi) # find the degree of wi in term
            # substitute (in the term) wi with E[wi^deg]^(1/w_deg)
            # so that the wi^w_deg term becomes (E[wi^deg]^(1/w_deg))^w_deg = E[wi^deg]
            term = subs(term, wi=>moments(w_deg)^(1/w_deg))
        end
        E_Bf += term
    end
    return E_Bf
end

# standard stochastic barrier certificates
function bc_standard(deg)
    # synthesize BC by using the standard formulation
    # deg: degree of BC template
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @polyvar w[1:1]
    f = dyn(vars,w[1]) #dyn_n(dyn,1)(vars,w)
    B, Bc, Bb = add_poly!(model, vars, deg) # generate polynomial template with given variables and degree
    Bf = B(vars=>f) #B(f(x,w)) https://juliapackages.com/p/dynamicpolynomials
    E_Bf = replaceW(Bf,w) # see noise terms with expectation

    @variable(model, 0 <= γ <= 1) # create the γ objective variable, γ between 0 and 1
    # 0th condition for B, >= 0 by default, div(deg+sos_tol,2) -> to make max degree of overall SOS polynomial even (so B can have odd degrees)
    model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 1st condition
    model,_ = add_psatz!(model, γ-B, vars, go, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true) #-B >= 0
    # 2nd condition
    model,_ = add_psatz!(model, B-1, vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 3rd and last condition
    model,_ = add_psatz!(model, B-E_Bf, vars, g, [], div(maxdegree(B-E_Bf)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)

    @objective(model, Min, γ)  # minimize γ
    optimize!(model) # solve for coefficients
    status = termination_status(model)
    Bc = value.(Bc)  # get the values of each coefficient
    for i in eachindex(Bc)
        Bc[i] = round(Bc[i]; digits = error) # round to order of error
    end
    # status might be optimal but if all Bc approx 10^{-error}, it's essentially 0 so check carefully.
    return (objective_value(model),status,Bc'*Bb) # optimal γ, optimization status and barrier certificate function
end

# ell = 0 -> bc_standard
function ibc(deg,ell,α)
    # synthesize IBC by using the given conditions
    # deg: degree of BC template
    # ell: max # BCs
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @polyvar w[1:1] # [1:1] makes it a vector form
    f = dyn(vars,w[1]) #dyn_n(dyn,1)(vars,w)
    # julia 1-indexing, i = 1 (i = 0 in paper)
    B, Bc, Bb = add_poly!(model, vars, deg) # generate polynomial template with given variables and degree
    Bf = B(vars=>f) # B(f(x)) https://juliapackages.com/p/dynamicpolynomials
    E_Bf = replaceW(Bf,w) # expectation of B(f(x,w))

    @variable(model, 0 <= γ <= 1) # minimization objective variable
    # 0th condition for all B >= 0
    model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 1st (exclusive) condition for B_0 <= γ ==> γ - B_0 ≥ 0
    model,_ = add_psatz!(model, γ-B, vars, go, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 2nd condition for B_0
    model,_ = add_psatz!(model, B-1 , vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)

    Bs  = [B] # store all unknown polynomial B(x) templates
    Bcs = [Bc] # store all corresponding unknown polynomial coefficients
    Bfs = [Bf] # store all corresponding unknown B(f(x)) templates
    E_Bfs = [E_Bf] # store all corresponding unknown E[B(f(x))] templates
    for i = 2:ell+1 # julia 1-indexing -> 2 <= i <= ell+1 (1 <= i <= ell in paper)
        B, Bc, Bb = add_poly!(model, vars, deg) 
        Bf = B(vars=>f)
        E_Bf = replaceW(Bf,w)

        push!(Bs, B) # append to Bs (.append equivalent), ! shows Bs will be modified inplace
        push!(Bcs, Bc)
        push!(Bfs, Bf)
        push!(E_Bfs, E_Bf)

        # 0th condition for all B_i
        model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
        # 2nd condition for B_i
        model,_ = add_psatz!(model, B-1 , vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
        # 3rd condition for B_i(x) = Bs[i-1], B_(i+1)(f(x)) = Bf
        model,_ = add_psatz!(model, α*Bs[i-1]-E_Bf, vars, g, [], div(maxdegree(Bs[i-1]-E_Bf)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    end
    # 4th and last exclusive condition for B_ell
    model,_ = add_psatz!(model, Bs[ell+1]-E_Bfs[ell+1], vars, g, [], div(maxdegree(Bs[ell+1]-E_Bfs[ell+1])+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    
    @objective(model, Min, γ) # minimize γ
    optimize!(model) #solve for coefficients
    status = termination_status(model)  #all_variables(model)
    IBC = [] # store B_i
    for i in eachindex(Bcs)
        Bc = value.(Bcs[i]) # get the values of each coefficient for B_i
        for j in eachindex(Bc)
            Bc[j] = round(Bc[j]; digits = error) # round to order of error
        end
        push!(IBC, Bc'*Bb)
    end

    objv = objective_value(model) # γ
    dual = dual_objective_value(model) # dual of SDP primal problem
    # println(deg," ",ell)
    # @show (objv - dual)   # difference between primal and dual objective
    # if difference is very small and status is potentially SLOW_PROGRESS
    if (abs(objv-dual) <= threshold && status != "OPTIMAL" && status != "INFEASIBLE")
        status = "optimal" # lower case to distinguish when this condition is executed
    end

    # status might be optimal but if all Bc approx 10^{-error}, it's essentially 0.
    return (objv,status,IBC) # optimal γ, optimization status and IBCs
end

# k = 1 -> bc_standard
function kbc(deg,k,c) # standard k-BC for safety from HSCC (Mahathi Anand et al)
    # synthesize k-Inductive BC by using the standard old formulation (not relaxed)
    # deg: degree of BC template
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @polyvar w[1:k] # k steps -> k iid noise terms
    f = dyn(vars,w[1])
    B, Bc, Bb = add_poly!(model, vars, deg) # generate polynomial template with given variables and degree
    Bf = B(vars=>f) #B(f(x,w)) https://juliapackages.com/p/dynamicpolynomials
    E_Bf = replaceW(Bf,w[1:1]) # expectation of B(f(x,w))

    fk = dyn_n(dyn,k)(vars,w) # f^k(x,w)
    Bfk = B(vars=>fk) # B(f^k(x,w))
    E_Bfk = replaceW(Bfk,w) # expectation of B(f^k(x,w))

    @variable(model, 0 <= γ <= 1)
    # 0th condition for all B, >= 0 by default
    model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 1st condition
    model,_ = add_psatz!(model, γ-B, vars, go, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true) #-B >= 0
    # 2nd condition
    model,_ = add_psatz!(model, B-1, vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    if k > 1 # k-induction steps
        # 3rd condition
        model,_ = add_psatz!(model, B+c-E_Bf, vars, g, [], div(maxdegree(B+c-E_Bf)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    end
    # 4th condition
    model,_ = add_psatz!(model, B-E_Bfk, vars, g, [], div(maxdegree(B-E_Bfk)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    
    @objective(model, Min, γ) # optimization objective
    optimize!(model) # solve for coefficients
    status = termination_status(model)
    Bc = value.(Bc)  # get the values of each coefficient
    for i in eachindex(Bc)
        Bc[i] = round(Bc[i]; digits = error) # round to order of error
    end

    objv = objective_value(model) # γ
    dual = dual_objective_value(model) # dual of SDP optimization
    # println(deg," ",k) 
    # @show (objv - dual) # difference between primal and dual optimal objective
    if (abs(objv-dual) <= threshold && status != "OPTIMAL" && status != "INFEASIBLE")
        status = "optimal"
    end

    # # status might be optimal but if all Bc approx 10^{-error}, it's essentially 0.
    return (objv,status,Bc'*Bb) # optimal γ, optimization status and k-BCs
end

# ell = 0, k = 1 -> bc_standard
# ell = 0 -> kbc
# k = 1 -> ibc
function kibc1(deg,ell,k,α,c) # k-IBC v1
    # synthesize IBC by using the given conditions
    # deg: degree of BC template
    # ell: max # BCs
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @polyvar w[1:k]
    f = dyn(vars,w[1]) #dyn_n(dyn,1)(vars,w)
    # julia 1-indexing, i = 1 (i = 0 in paper)
    B, Bc, Bb = add_poly!(model, vars, deg) # generate polynomial template with given variables and degree
    Bf = B(vars=>f) #B(f(x)) https://juliapackages.com/p/dynamicpolynomials
    E_Bf = replaceW(Bf,w[1:1]) # expectation of B(f(x,w))

    @variable(model, 0 <= γ <= 1) # optimization objective
    # 0th condition for all B >= 0
    model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 1st (exclusive) condition for B_0
    model,_ = add_psatz!(model, γ-B, vars, go, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 2nd condition for B_0
    model,_ = add_psatz!(model, B-1 , vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)

    Bs  = [B] # store all unknown polynomial B(x) templates
    Bcs = [Bc] # store all corresponding unknown polynomial coefficients
    Bfs = [Bf] # store all corresponding unknown B(f(x)) templates
    E_Bfs = [E_Bf] # store all corresponding unknown E[B(f(x))] templates
    for i = 2:ell+1 # julia 1-indexing -> 2 <= i <= ell+1 (1 <= i <= ell in paper)
        B, Bc, Bb = add_poly!(model, vars, deg) 
        Bf = B(vars=>f)
        E_Bf = replaceW(Bf,w[1:1]) # expectation of B(f(x,w))

        push!(Bs, B) # append to Bs in python
        push!(Bcs, Bc)
        push!(Bfs, Bf)
        push!(E_Bfs, E_Bf)

        # 0th condition for all B
        model,_ = add_psatz!(model, B, vars, g, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
        # 2nd condition for B_i
        model,_ = add_psatz!(model, B-1 , vars, gu, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
        # 3rd condition for B_i(x) = Bs[i-1], B_(i+1)(f(x)) = Bf
        model,_ = add_psatz!(model, α*Bs[i-1]-E_Bf, vars, g, [], div(maxdegree(Bs[i-1]-E_Bf)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    end

    if k>1
        # 4th condition, exclusive condition for B_ell
        model,_ = add_psatz!(model, Bs[ell+1]+c-E_Bfs[ell+1], vars, g, [], div(maxdegree(Bs[ell+1]+c-E_Bfs[ell+1])+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    end

    fk = dyn_n(dyn,k)(vars,w) #f^k(x,w)
    Bfk = (Bs[ell+1])(vars=>fk) # B_ell(f^k(x,w))
    E_Bfk = replaceW(Bfk,w) # expectation of B_ell(f^k(x,w))

    # 5th condition and last exclusive condition for B_ell
    model,_ = add_psatz!(model, Bs[ell+1]-E_Bfk, vars, g, [], div(maxdegree(Bs[ell+1]-E_Bfk)+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    
    @objective(model, Min, γ) # optimization objective
    optimize!(model) #solve for coefficients
    status = termination_status(model)  #all_variables(model)
    kIBC = [] # store all B_i

    for i in eachindex(Bcs)
        Bc = value.(Bcs[i]) # get the values of each coefficient for B_i
        for j in eachindex(Bc)
            Bc[j] = round(Bc[j]; digits = error) # round to order of error
        end
        push!(kIBC, Bc'*Bb)
    end

    objv = objective_value(model) # primal optimal objective
    dual = dual_objective_value(model) # dual optimal objective
    # println(deg," ",ell," ",k)
    @show (objv - dual)  # difference between primal and dual optimal objective
    if (abs(objv-dual) <= threshold && status != "OPTIMAL" && status != "INFEASIBLE")
        status = "optimal"
    end

    # status might be optimal but if all Bc approx 10^{-error}, it's essentially 0.
    return (objv,status,kIBC)
end


# Simulation
names = ["simple", "lotka_volt", "logistic_map"] # put names of systems here from systems folder  
system_data = ["vars: ","f: ","go: ","gu: ","g: ", "alphas: ", "c: "]
max_deg, ell_max, k_max = 7, 2, 3
for i in eachindex(names)
    name = names[i]
    include("./systems/"*name*".jl"); # load system dynamics info
    file = open("./systems/"*name*"_system.txt", "w"); # open file to write/save system dynamics info
    @polyvar w; 
    f = dyn(vars,w) # f(x,w)

    for (j,m) in zip(system_data, [vars,f,go,gu,g,α[i],c[i]]) # save these data in readable form
        write(file, j*"{")
        for i = 1:length(m)-1
            write(file, string(m[i])*", ")
        end
        write(file, string(last(m))*"}\n")
    end
    close(file)

    if name == "simple" || name == "lotka_volt"
        # print sufficient condition results for standard stochastic barrier certificate
        file = open("./systems/"*name*"_bc.txt", "w");
        for deg = 1:max_deg
            stats = @timed data = bc_standard(deg) # time execution
            gamma, status, B = data
            write(file, "poly deg: "*string(deg)*"\n")
            write(file, "status: "*string(status)*",\t γ(min): "*string(gamma)*"\n")
            write(file, Base.replace(string(B),"e"=>"*10^")*"\n")
            write(file, "time: "*string(stats.time)*"\n\n") 
        end
        close(file)
    
        # print IBC condition results
        file = open("./systems/"*name*"_ibc.txt", "w");
        for deg = 1:5
            for ell = 0:ell_max
                stats = @timed data = ibc(deg, ell, α[i])
                gamma, status, IBC = data
                write(file, "poly deg: "*string(deg)*",\t ell: "*string(ell)*"\n")
                write(file, "status: "*string(status)*",\t γ(min): "*string(gamma)*"\n")
                write(file, Base.replace(string(IBC),"e"=>"*10^")*"\n")
                write(file, "time: "*string(stats.time)*"\n\n")
            end
        end
        close(file)
    end

    # print k-Inductive BC condition results
    if name == "simple"
        file = open("./systems/"*name*"_kbc.txt", "w");
        for deg = 3:max_deg+2
            for k = 1:k_max
                stats = @timed data = kbc(deg, k, c[i])
                gamma, status, kBC = data
                write(file, "poly deg: "*string(deg)*",\t k: "*string(k)*"\n")
                write(file, "status: "*string(status)*",\t γ(min): "*string(gamma)*"\n")
                write(file, Base.replace(string(kBC),"e"=>"*10^")*"\n")
                write(file, "time: "*string(stats.time)*"\n\n")
            end
        end
        close(file)
    end

    # print k-IBC v1 condition results
    if name == "logistic_map" || name == "pp4state"
        file = open("./systems/"*name*"_kibc1.txt", "w");
        for deg = 3:8
            println("degree: ", deg)
            for ell = 0:2
                for k = 1:2
                    # takes too long under these conditions
                    if(name == "pp4state" && deg >= 5 && (ell > 0 || k > 1))
                        break
                    end
                    stats = @timed data = kibc1(deg, ell, k, α[i], c[i])
                    gamma, status, kIBC = data
                    write(file, "poly deg: "*string(deg)*",\t ell: "*string(ell)*",\t k: "*string(k)*"\n")
                    write(file, "status: "*string(status)*",\t γ(min): "*string(gamma)*"\n")
                    write(file, Base.replace(string(kIBC),"e"=>"*10^")*"\n")
                    write(file, "time: "*string(stats.time)*"\n\n")
                end
            end
        end
        close(file)
    end

    println("Finished "*name)
end

