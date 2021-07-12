using Test
using Plots
include("robin-interface-cylindrical-solution.jl")
RAS = RobinAnalyticalSolution

k1 = 1.0
k2 = 2.0
q1 = 2.0
q2 = 1.0
lambda = 1.0
innerradius = 0.3
outerradius = 1.0
Tm = 0.5
Tw = 1.0

solver = RAS.CylindricalSolver(
    q1,
    k1,
    q2,
    k2,
    lambda,
    innerradius,
    outerradius,
    Tw,
    Tm,
)

Touter = RAS.analytical_solution(outerradius,solver)

TR1 = RAS.core_solution(innerradius,q1,k1,solver.B1)
TR2 = RAS.rim_solution(innerradius,q2,k2,solver.A2,solver.B2)
@test TR1 ≈ TR2

k1dT1 = k1*RAS.core_radial_derivative(innerradius,q1,k1)
k2dT2 = k2*RAS.rim_radial_derivative(innerradius,q2,k2,solver.A2)

flux = k1dT1 - k2dT2 - lambda*(Tm - TR1)

# lhs = k2/innerradius*solver.A2 - lambda*solver.B1
# rhs = 0.5*(q2-q1)*innerradius - lambda*(q1/4k1*innerradius^2 + Tm)

# m = RAS.coefficient_matrix(k2,lambda,innerradius,outerradius)
# r = RAS.coefficient_rhs(q1,k1,q2,k2,lambda,innerradius,outerradius,Tw,Tm)
#
# B1, A2, B2 = m\r
#
# lhs = k2/innerradius*A2 - lambda*B1
