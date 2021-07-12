module AnalyticalSolver

struct Analytical1D
    q1::Any
    k1::Any
    q2::Any
    k2::Any
    lambda::Any
    R::Any
    T1::Any
    T2::Any
    A1::Any
    B1::Any
    A2::Any
    B2::Any
    function Analytical1D(q1, k1, q2, k2, lambda, R, T1, T2, Tm)
        m = analytical_coefficient_matrix(k1, k2, lambda, R)
        r = analytical_coefficient_rhs(q1, k1, q2, k2, lambda, R, T1, T2, Tm)
        B1 = T1
        A1, A2, B2 = m \ r
        new(q1, k1, q2, k2, lambda, R, T1, T2, A1, B1, A2, B2)
    end
end

function analytical_coefficient_matrix(k1, k2, lambda, R)
    m = zeros(3, 3)
    m[1, 2:3] .= 1.0
    m[2, 1] = k1 + lambda * R
    m[2, 2] = -k2
    m[3, 1] = -R
    m[3, 2] = R
    m[3, 3] = 1.0
    return m
end

function analytical_coefficient_rhs(q1, k1, q2, k2, lambda, R, T1, T2, Tm)
    r = zeros(3)
    r[1] = T2 + q2 / 2k2
    r[2] = R * (q1 - q2) + lambda * (Tm - T1) + lambda * q1 / 2k1 * R^2
    r[3] = T1 + R^2 / 2 * (q2 / k2 - q1 / k1)
    return r
end

function evaluate_solution(x, q, k, A, B)
    return -q / 2k * x^2 + A * x + B
end

function left_solution(x, solver::Analytical1D)
    q1, k1, A1, B1 = solver.q1, solver.k1, solver.A1, solver.B1
    return evaluate_solution(x, q1, k1, A1, B1)
end

function right_solution(x, solver::Analytical1D)
    q2, k2, A2, B2 = solver.q2, solver.k2, solver.A2, solver.B2
    return evaluate_solution(x, q2, k2, A2, B2)
end

# function (solver::Analytical1D)(x)
#     R = solver.R
#
#     if x < R
#         return left_solution(x, solver)
#     else
#         return right_solution(x, solver)
#     end
# end

function evaluate_gradient(x, q, k, A)
    return -q / k * x + A
end

function left_gradient(x, solver::Analytical1D)
    q1, k1, A1 = solver.q1, solver.k1, solver.A1
    return evaluate_gradient(x, q1, k1, A1)
end

function right_gradient(x, solver::Analytical1D)
    q2, k2, A2 = solver.q2, solver.k2, solver.A2
    return evaluate_gradient(x, q2, k2, A2)
end

function solution_gradient(x, solver::Analytical1D)
    R = solver.R
    if x < R
        return left_gradient(x, solver)
    else
        return right_gradient(x, solver)
    end
end

# end module
end
# end module
