using LinearAlgebra
using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("DG1D.jl")
include("../useful_routines.jl")
include("../analytical_solution_1d.jl")
AS = AnalyticalSolver


function measure_error(
    ne,
    solverbasis,
    quad,
    q1,
    k1,
    q2,
    k2,
    lambda,
    interfacepoint,
    TL,
    TR,
    Tm,
    penaltyfactor,
    exactsolution,
)
    refpoints = interpolation_points(solverbasis)
    mesh = DG1D.DGMesh1D(0.0, 1.0, interfacepoint, ne, ne, refpoints)
    minelmtsize = minimum(DG1D.element_size(mesh))
    penalty = penaltyfactor / minelmtsize

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    DG1D.assemble_gradient_operator!(sysmatrix, solverbasis, quad, k1, k2, mesh)
    DG1D.assemble_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
    DG1D.assemble_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
    DG1D.assemble_interface_robin_operator!(
        sysmatrix,
        solverbasis,
        lambda,
        mesh,
    )
    DG1D.assemble_interface_robin_source!(sysrhs, solverbasis, lambda, Tm, mesh)
    DG1D.assemble_two_phase_source!(
        sysrhs,
        x -> q1,
        x -> q2,
        solverbasis,
        quad,
        mesh,
    )
    DG1D.assemble_boundary_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
    DG1D.assemble_boundary_penalty_operator!(
        sysmatrix,
        solverbasis,
        penalty,
        mesh,
    )
    DG1D.assemble_boundary_rhs!(sysrhs, k1, k2, TL, TR, solverbasis, penalty, mesh)

    matrix = DG1D.sparse_operator(sysmatrix, mesh, 1)
    rhs = DG1D.rhs_vector(sysrhs, mesh, 1)

    solution = matrix \ rhs

    err =
        uniform_mesh_L2_error(solution', exactsolution, solverbasis, quad, mesh)
    den = integral_norm_on_uniform_mesh(exactsolution, quad, mesh, 1)

    symmflag = issymmetric(matrix)

    return err[1] / den[1], symmflag
end


################################################################################
q1, q2 = 1.0, 0.5
k1, k2 = 0.5, 1.0
lambda = 1.0
interfacepoint = 0.7
TL = 0.0
TR = 1.0
Tm = 0.3

exactsolution =
    AS.Analytical1D(q1, k1, q2, k2, lambda, interfacepoint, TL, TR, Tm)

penalty = 1e2
solverorder = 1
solverbasis = LagrangeTensorProductBasis(1, solverorder)
numqp = required_quadrature_order(solverorder)
quad = tensor_product_quadrature(1, numqp)

powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers

rvals = [
    measure_error(
        ne,
        solverbasis,
        quad,
        q1,
        k1,
        q2,
        k2,
        lambda,
        interfacepoint,
        TL,
        TR,
        Tm,
        penalty,
        x -> exactsolution(x[1]),
    ) for ne in nelmts
]
err1 = [r[1] for r in rvals]
flags1 = [r[2] for r in rvals]

maxdomainsize = max(interfacepoint,1-interfacepoint)
dx = maxdomainsize ./ nelmts

rate1 = convergence_rate(dx, err1)
@test all(rate1 .> 1.95)
@test all(flags1)
################################################################################

################################################################################
solverorder = 2
solverbasis = LagrangeTensorProductBasis(1, solverorder)
numqp = required_quadrature_order(solverorder)
quad = tensor_product_quadrature(1, numqp)

rvals = [measure_error(
    ne,
    solverbasis,
    quad,
    q1,
    k1,
    q2,
    k2,
    lambda,
    interfacepoint,
    TL,
    TR,
    Tm,
    penalty,
    x -> exactsolution(x[1]),
) for ne in nelmts]

err2 = [r[1] for r in rvals]
flags2 = [r[2] for r in rvals]

@test all(err2 .< 1e6eps())
@test all(flags2)
################################################################################


using DataFrames, CSV
df = DataFrame(ElementSize = dx, linear = err1, quadratic = err2)

foldername = "DG1D\\robin-interface-tests\\"
filename = foldername*"robin-convergence.csv"
# CSV.write(filename,df)
