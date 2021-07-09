using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../interior_penalty.jl")
include("../../useful_routines.jl")

function exact_solution(v)
    x, y = v
    return cos(2pi * x) * sin(2pi * y)
end

function source_term(v, k)
    x, y = v
    return 8pi^2 * k * cos(2pi * x) * sin(2pi * y)
end

function measure_error(
    nelmts,
    solverorder,
    levelsetorder,
    distancefunction,
    sourceterm,
    exactsolution,
    k1,
    k2,
    penaltyfactor,
)
    numqp = required_quadrature_order(solverorder)
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
        sourceterm,
        x -> exactsolution(x),
        solverbasis,
        cellquads,
        facequads,
        k1,
        k2,
        penalty,
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
    ################################################################################

    return err[1]
end

################################################################################
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1e3
distancefunction(x) = ones(size(x)[2])

err1 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts
rate1 = convergence_rate(dx, err1)
pushfirst!(rate1, 0.0)
################################################################################



################################################################################
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 2
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1e3
distancefunction(x) = ones(size(x)[2])

err2 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        k1,
        k2,
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

foldername = "InteriorPenalty\\uncut_mesh_tests\\"
filename = foldername * "IP_convergence.csv"
# CSV.write(filename,df)
################################################################################
