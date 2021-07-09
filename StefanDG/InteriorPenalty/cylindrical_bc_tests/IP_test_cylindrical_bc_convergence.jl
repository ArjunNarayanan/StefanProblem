using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../interior_penalty.jl")
include("../../cylinder-analytical-solution.jl")
include("../../useful_routines.jl")


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

function (solver::AnalyticalSolution.CylindricalSolver)(x, center)
    dx = x - center
    r = sqrt(dx' * dx)
    return AnalyticalSolution.analytical_solution(r, solver)
end


################################################################################
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
numqp = required_quadrature_order(solverorder)
levelsetorder = 2
k1 = 1.0
k2 = 2.0
q1 = 1.0
q2 = 2.0
penaltyfactor = 1e3
center = [0.0, 0.0]
innerradius = 0.5
outerradius = 1.5
Tw = 1.0
distancefunction(x) = circle_distance_function(x, center, innerradius)
analyticalsolution = AnalyticalSolution.CylindricalSolver(
    q1,
    k1,
    q2,
    k2,
    innerradius,
    outerradius,
    Tw,
)

err1 = [
    measure_error(
        ne,
        solverorder,
        numqp,
        levelsetorder,
        distancefunction,
        x -> q1,
        x -> q2,
        x -> analyticalsolution(x, center),
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate1 = convergence_rate(dx, err1)
pushfirst!(rate1,0.0)
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
        x -> analyticalsolution(x, center),
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate2 = convergence_rate(dx, err2)
pushfirst!(rate2,0.0)
# ################################################################################





################################################################################
using DataFrames, CSV
df = DataFrame(
    NElmts = nelmts,
    linear = err1,
    rate1 = rate1,
    quadratic = err2,
    rate2 = rate2,
)

foldername = "InteriorPenalty\\cylindrical_bc_tests\\"
filename = foldername * "IP_convergence.csv"
# CSV.write(filename,df)
################################################################################
