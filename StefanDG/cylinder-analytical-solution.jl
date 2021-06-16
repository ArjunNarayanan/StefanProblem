module AnalyticalSolution

struct CylindricalSolver
    q::Any
    k1::Any
    k2::Any
    R::Any
    b::Any
    Tw::Any
    function CylindricalSolver(q, k1, k2, R, b, Tw)
        new(q, k1, k2, R, b, Tw)
    end
end

function solution_field(r, q, k2, R, b, Tw)
    T = Tw + q / 4k2 * ((b^2 - r^2) + 2R^2 * (log(r) - log(b)))
    return T
end

function analytical_solution(r, q, k2, R, b, Tw)
    if r < R
        return solution_field(R, q, k2, R, b, Tw)
    else
        return solution_field(r, q, k2, R, b, Tw)
    end
end

function analytical_solution(r, solver::CylindricalSolver)
    return analytical_solution(
        r,
        solver.q,
        solver.k2,
        solver.R,
        solver.b,
        solver.Tw,
    )
end

function analytical_radial_derivative(r, q, k2, R)
    if r < R
        return 0.
    else
        return Tr = -q / 2k2 * r * (1.0 - R^2 / r^2)
    end
end

function analytical_radial_derivative(r, solver::CylindricalSolver)
    return analytical_radial_derivative(r, solver.q, solver.k2, solver.R)
end

end
