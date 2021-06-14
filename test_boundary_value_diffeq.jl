using Plots
using BoundaryValueDiffEq

b = 1.0
R = 0.8
Tw = 1.0
k1 = k2 = 1.0
q = 1.0

function right_hand_side!(dT, T, p, r)
    q, k2, R, b, Tw = p
    if r < R
        return dT[1] = 0.0
    else
        return dT[1] = -q / k2 * r
    end
end

function wall_bc!(residual, T, p, r)
    q, k2, R, b, Tw = p
    residual[1] = T[end][1] - Tw
end

p = [q, k2, R, b, Tw]
bvp = BVProblem(right_hand_side!, wall_bc!, [0.], (0.0, 1.0), p)
sol = solve(bvp)
plot(sol)
