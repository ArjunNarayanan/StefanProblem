module AnalyticalSolution

struct CylindricalSolver
    q::Any
    k1::Any
    k2::Any
    R::Any
    b::Any
    Tw::Any
    B1::Any
    A2::Any
    B2::Any
    function CylindricalSolver(q, k1, k2, R, b, Tw)
        B1, A2, B2 = analytical_coefficients(q, k2, R, b, Tw)
        new(q, k1, k2, R, b, Tw, B1, A2, B2)
    end
end

function analytical_coefficients(q, k2, R, b, Tw)
    B1 = Tw + q / 4k2 * (b^2 - R^2 + 2R^2 * (log(R) - log(b)))
    A2 = q / k2 * R^2 / 2
    B2 = Tw + q / 4k2 * (b^2 - 2R^2 * log(b))

    return [B1, A2, B2]
end

function analytical_solution(r, q, k2, R, b, B1, A2, B2)
    if r < R
        return B1
    else
        T = -q / k2 * r^2 / 4 + A2 * log(r) + B2
        return T
    end
end

function analytical_solution(r, solver::CylindricalSolver)
    return analytical_solution(
        r,
        solver.q,
        solver.k2,
        solver.R,
        solver.b,
        solver.B1,
        solver.A2,
        solver.B2,
    )
end

end
