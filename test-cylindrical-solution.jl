using Plots
using DifferentialEquations

include("cylinder-analytical-solution.jl")
AS = AnalyticalSolution

b = 1.0
R = 0.8
Tw = 1.0
k1 = k2 = 1.0
q = 1.0

B1, A2, B2 = AS.analytical_coefficients(q, k2, R, b, Tw)

# plot(r->AS.analytical_solution(r,q,k2,R,b,B1,A2,B2),0,1)

function right_hand_side(T, p, r)
    q, k2, R, b, Tw = p
    if r < R
        return 0.0
    else
        return -q / k2 * r
    end
end

function wall_bc!(residual, T, p, r)
    q, k2, R, b, Tw = p
    residual[1] = T[end] - Tw
end

p = [q, k2, R, b, Tw]
bvp = BVProblem(right_hand_side, wall_bc!, p, (0.0, 1.0))
sol = solve(bvp)
