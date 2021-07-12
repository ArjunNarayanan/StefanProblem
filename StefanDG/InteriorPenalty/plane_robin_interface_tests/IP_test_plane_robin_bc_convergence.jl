using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../interior_penalty.jl")
include("../../useful_routines.jl")
include("analytical_solution_1d.jl")


function measure_error(
    nelmts,
    solverorder,
    numqp,
    levelsetorder,
    distancefunction,
    rhsfunc1,
    rhsfunc2,
    exactsolution,
    k1,
    k2,
    lambda,
    Tm,
    penaltyfactor,
)

    solverbasis = LagrangeTensorProductBasis(2, solverorder)
    levelsetbasis = LagrangeTensorProductBasis(2, levelsetorder)

    dgmesh = CutCellDG.DGMesh(
        [0.0, 0.0],
        [1.0, 1.0],
        [nelmts, nelmts],
        interpolation_points(solverbasis),
    )
    cgmesh = CutCellDG.CGMesh(
        [0.0, 0.0],
        [1.0, 1.0],
        [nelmts, nelmts],
        number_of_basis_functions(levelsetbasis),
    )
    levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis)
    minelmtsize = minimum(CutCellDG.element_size(dgmesh))
    penalty = penaltyfactor / minelmtsize

    cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

    mergedmesh =
        CutCellDG.MergedMesh!(cutmesh, cellquads, facequads, interfacequads)


    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    InteriorPenalty.assemble_interior_penalty_linear_system!(
        sysmatrix,
        solverbasis,
        cellquads,
        facequads,
        interfacequads,
        k1,
        k2,
        penalty,
        mergedmesh,
    )
    InteriorPenalty.assemble_robin_operator!(
        sysmatrix,
        solverbasis,
        interfacequads,
        lambda,
        mergedmesh,
    )
    InteriorPenalty.assemble_interior_penalty_rhs!(
        sysrhs,
        x -> 0.0,
        exactsolution,
        solverbasis,
        cellquads,
        facequads,
        k1,
        k2,
        penalty,
        mergedmesh,
    )
    InteriorPenalty.assemble_two_phase_source!(
        sysrhs,
        rhsfunc1,
        rhsfunc2,
        solverbasis,
        cellquads,
        mergedmesh,
    )
    InteriorPenalty.assemble_robin_source!(
        sysrhs,
        solverbasis,
        interfacequads,
        lambda,
        Tm,
        mergedmesh,
    )
    ################################################################################
    matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
    @assert issymmetric(matrix)
    rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 1)
    sol = matrix \ rhs
    ################################################################################

    ################################################################################
    err = mesh_L2_error(sol', exactsolution, solverbasis, cellquads, mergedmesh)
    den = integral_norm_on_mesh(exactsolution, cellquads, mergedmesh, 1)
    ################################################################################


    return err[1] / den[1]
end

function (solver::AnalyticalSolver.Analytical1D)(x, R)
    if x[1] < R
        return AnalyticalSolver.left_solution(x[1], solver)
    else
        return AnalyticalSolver.right_solution(x[1], solver)
    end
end


################################################################################
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
numqp = required_quadrature_order(solverorder) + 2
penaltyfactor = 1e3
levelsetorder = 2
k1 = 1.0
k2 = 2.0
q1 = 2.0
q2 = 1.0
lambda = 1.0
Tm = 0.5
interfacepoint = 0.7
TL = 0.0
TR = 1.0

analyticalsolution = AnalyticalSolver.Analytical1D(
    q1,
    k1,
    q2,
    k2,
    lambda,
    interfacepoint,
    TL,
    TR,
    Tm,
)

# using Plots
# xrange = 0:1e-2:1
# plot(xrange,x->analyticalsolution(x,interfacepoint))

distancefunction(x) =
    plane_distance_function(x, [-1.0, 0.0], [interfacepoint, 0.0])

err1 = [
    measure_error(
        ne,
        solverorder,
        numqp,
        levelsetorder,
        distancefunction,
        x -> q1,
        x -> q2,
        x -> analyticalsolution(x, interfacepoint),
        k1,
        k2,
        lambda,
        Tm,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate1 = convergence_rate(dx, err1)
pushfirst!(rate1, 0.0)
################################################################################


################################################################################
solverorder = 2
numqp = required_quadrature_order(solverorder) + 2

err2 = [
    measure_error(
        ne,
        solverorder,
        numqp,
        levelsetorder,
        distancefunction,
        x -> q1,
        x -> q2,
        x -> analyticalsolution(x, interfacepoint),
        k1,
        k2,
        lambda,
        Tm,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate2 = convergence_rate(dx, err2)
pushfirst!(rate2, 0.0)
################################################################################





################################################################################
using DataFrames, CSV
df = DataFrame(
    NElmts = nelmts,
    linear = err1,
    rate1 = rate1,
    quadratic = err2,
    rate2 = rate2,
)

# foldername = "InteriorPenalty\\plane_robin_interface_tests\\"
# filename = foldername * "IP_convergence.csv"
# CSV.write(filename,df)
# ################################################################################
