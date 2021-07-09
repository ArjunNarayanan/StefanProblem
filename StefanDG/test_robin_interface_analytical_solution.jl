using Test
using Plots
include("cylinder-analytical-solution.jl")
AS = AnalyticalSolution
include("robin-interface-cylindrical-solution.jl")
RAS = RobinAnalyticalSolution


q1, q2 = 1.,2.
k1,k2 = 2.,1.
lambda = 0.0
R = 0.8
b = 1.0
Tw = 1.
Tm = 2.

s1 = AS.CylindricalSolver(q1,k1,q2,k2,R,b,Tw)
s2 = RAS.CylindricalSolver(q1,k1,q2,k2,lambda,R,b,Tw,Tm)

rvals = 0:1e-2:1

v1 = [AS.analytical_solution(r,s1) for r in rvals]
v2 = [RAS.analytical_solution(r,s2) for r in rvals]

@test all(v1 .≈ v2)
