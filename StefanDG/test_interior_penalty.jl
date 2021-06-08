using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("interior_penalty.jl")
include("useful_routines.jl")

function exact_solution(v)
    x, y = v
    return cos(2pi * x) * sin(2pi * y)
end

function source_term(v, k)
    x, y = v
    return -8pi^2 * k * cos(2pi * x) * sin(2pi * y)
end

nelmts = 16
solverorder = 3
levelsetorder = 1
distancefunction(x) = plane_distance_function(x, [1.0, 0.0], [0.5, 0.0])


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
    levelset = CutCellDG.LevelSet(x -> ones(size(x)[2]), cgmesh, levelsetbasis)
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
        x->[exactsolution(x)],
        solverbasis,
        cellquads,
        facequads,
        penalty,
        mergedmesh,
    )
    ################################################################################
    matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
    rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 1)
    sol = matrix \ rhs
    ################################################################################

    ################################################################################
    err = mesh_L2_error(sol', exactsolution, solverbasis, cellquads, mergedmesh)
    ################################################################################

    return err[1]
end

################################################################################
powers = [2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1e3
distancefunction(x) = plane_distance_function(x, [1.0, 0.0], [0.5, 0.0])

err = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> [source_term(x, k1)],
        exact_solution,
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts
rate = convergence_rate(dx,err)
################################################################################



################################################################################
powers = [2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 2
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1e3
distancefunction(x) = plane_distance_function(x, [1.0, 0.0], [0.5, 0.0])

err = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> [source_term(x, k1)],
        exact_solution,
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts
rate = convergence_rate(dx,err)
################################################################################
