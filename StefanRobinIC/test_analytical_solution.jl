using Test
include("analytical_solution_1d.jl")
AS = AnalyticalSolver

q1, q2 = 1.0, 1.0
k1, k2 = 0.5, 0.1
lambda = 1.0
R = 0.5
T1 = 0.0
T2 = 1.0
Tm = 0.5

solver = AS.Analytical1D(q1, k1, q2, k2, lambda, R, T1, T2, Tm)

xrange = 0:1e-2:1


TR1 = AS.left_solution(R,solver)
TR2 = AS.right_solution(R,solver)
@test TR1 ≈ TR2

TL = AS.left_solution(0.,solver)
@test TL ≈ T1
TR = AS.right_solution(1.,solver)
@test TR ≈ T2

dT1 = AS.left_gradient(R,solver)
dT2 = AS.right_gradient(R,solver)

flux = k1*dT1 - k2*dT2 - lambda*(Tm - TR1)
@test flux ≈ 0.0


# T = solver.(xrange)
# using Plots
# plot(xrange,T)
