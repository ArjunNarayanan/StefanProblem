module AnalyticalSolution

struct CylindricalSolver
    q1::Any
    k1::Any
    q2::Any
    k2::Any
    R::Any
    b::Any
    Tw::Any
    function CylindricalSolver(q1, k1, q2, k2, R, b, Tw)
        new(q1, k1, q2, k2, R, b, Tw)
    end
end

function core_solution(r, q1, k1, q2, k2, R, b, Tw)
    T =
        Tw +
        q1 / 4k1 * (R^2 - r^2) +
        q2 / 4k2 * (b^2 - R^2) +
        R^2 / 2k2 * (q2 - q1) * (log(R) - log(b))
    return T
end

function rim_solution(r, q1, k1, q2, k2, R, b, Tw)
    T =
        Tw + q2 / 4k2 * (b^2 - r^2) + q2 / 2k2 * R^2 * (log(r) - log(b)) -
        q1 / 2k2 * R^2 * log(r)
end

function analytical_solution(r, q1, k1, q2, k2, R, b, Tw)
    if r < R
        return core_solution(r, q1, k1, q2, k2, R, b, Tw)
    else
        return rim_solution(r, q1, k1, q2, k2, R, b, Tw)
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
        solver.b,
        solver.Tw,
    )
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
    return analytical_radial_derivative(r, solver.q1, solver.k1,solver.q2, solver.k2, solver.R)
end

end
