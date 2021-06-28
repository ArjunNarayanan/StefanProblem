module AnalyticalSolution

struct CylindricalSolver
    q1::Any
    k1::Any
    q2::Any
    k2::Any
    R::Any
    b::Any
    Tw::Any
    B1::Any
    A2::Any
    B2::Any
    function CylindricalSolver(q1, k1, q2, k2, R, b, Tw)
        A2 = (q2 - q1) / (2k2) * R^2
        B2 = Tw + q2 / (4k2) * b^2 - A2 * log(b)
        B1 = R^2 * (q1 / k1 - q2 / k2) / 4 + B2 + A2 * log(R)

        new(q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    end
end

function core_solution(r, q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    T = -q1 / (4k1) * r^2 + B1
    return T
end

function rim_solution(r, q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    T = -q2 / (4k2) * r^2 + A2 * log(r) + B2
end

function analytical_solution(r, q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    if r < R
        return core_solution(r, q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    else
        return rim_solution(r, q1, k1, q2, k2, R, b, Tw, B1, A2, B2)
    end
end

function coefficients(solver::CylindricalSolver)
    return [
        solver.q1,
        solver.k1,
        solver.q2,
        solver.k2,
        solver.R,
        solver.b,
        solver.Tw,
        solver.B1,
        solver.A2,
        solver.B2,
    ]
end

function analytical_solution(r, solver::CylindricalSolver)
    return analytical_solution(r, coefficients(solver)...)
end

function analytical_radial_derivative(r, q1, k1, q2, k2, R)
    if r < R
        Tr = -q1 / 2k1 * r
        return Tr
    else
        Tr = -q1 / 2k2 * R^2 / r - q2 / 2k2 * r * (1 - R^2 / r^2)
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
