module RobinAnalyticalSolution

struct CylindricalSolver
    q1::Any
    k1::Any
    q2::Any
    k2::Any
    lambda::Any
    R::Any
    b::Any
    Tw::Any
    Tm::Any
    B1::Any
    A2::Any
    B2::Any
    function CylindricalSolver(q1, k1, q2, k2, lambda, R, b, Tw, Tm)
        m = coefficient_matrix(k2, lambda, R, b)
        r = coefficient_rhs(q1, k1, q2, k2, lambda, R, b, Tw, Tm)
        B1, A2, B2 = m \ r
        new(q1, k1, q2, k2, lambda, R, b, Tw, Tm, B1, A2, B2)
    end
end

function coefficient_matrix(k2, lambda, R, b)
    m = zeros(3, 3)
    m[1, 1:3] = [1.0, -log(R), -1.0]
    m[2, 1:3] = [0.0, k2 / R, -lambda]
    m[3, 1:3] = [0.0, log(b), 1.0]

    return m
end

function coefficient_rhs(q1, k1, q2, k2, lambda, R, b, Tw, Tm)
    r = zeros(3)
    r[1] = R^2 * (q1 / 4k1 - q2 / 4k2)
    r[2] = 0.5 * (q2 - q1) * R - lambda * (q1 / 4k1 * R^2 + Tm)
    r[3] = q2 / 4k2 * b^2 + Tw

    return r
end

function core_solution(r, q1, k1, B1)
    T = -q1 / (4k1) * r^2 + B1
    return T
end

function rim_solution(r, q2, k2, A2, B2)
    T = -q2 / (4k2) * r^2 + A2 * log(r) + B2
end

function analytical_solution(r, q1, k1, q2, k2, R, B1, A2, B2)
    if r < R
        return core_solution(r, q1, k1, B1)
    else
        return rim_solution(r, q2, k2, A2, B2)
    end
end

function analytical_solution(r, solver::CylindricalSolver)
    return analytical_solution(
        r,
        solver.q1,
        solver.k1,
        solver.q2,
        solver.k2,
        solver.R,
        solver.B1,
        solver.A2,
        solver.B2,
    )
end

function analytical_radial_derivative(r, q1, k1, q2, k2)
    if r < R
        Tr = -q1 / 2k1 * r
        return Tr
    else
        Tr = -q2 / 2k2 * r + A2 / r
        return Tr
    end
end

function analytical_radial_derivative(r, solver::CylindricalSolver)
    return analytical_radial_derivative(
        r,
        solver.q1,
        solver.k1,
        solver.q2,
        solver.k2,
        solver.R,
    )
end

end
