using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../interior_penalty.jl")
include("../../useful_routines.jl")

function exact_solution(v)
    x, y = v
    return 3x + 4y
end

function source_term(v, k)
    return 0.0
end

function measure_error(
    nelmts,
    solverorder,
    numqp,
    levelsetorder,
    distancefunction,
    sourceterm,
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

    # penalty = penaltyfactor
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
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
numqp = required_quadrature_order(solverorder)+2
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1e3

interfacepoint = [0.8,0.]
interfaceangle = 60.0
interfacenormal = [cosd(interfaceangle),sind(interfaceangle)]
distancefunction(x) = plane_distance_function(x,interfacenormal,interfacepoint)

err1 = [
    measure_error(
        ne,
        solverorder,
        numqp,
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
rate1 = convergence_rate(dx,err1)
################################################################################



################################################################################
solverorder = 2
numqp = required_quadrature_order(solverorder)+2

err2 = [
    measure_error(
        ne,
        solverorder,
        numqp,
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
rate2 = convergence_rate(dx,err2)
################################################################################



################################################################################
using DataFrames, CSV
df = DataFrame(NElmts = nelmts,linear = err1, quadratic = err2)

foldername = "InteriorPenalty\\plane_interface_tests\\"
filename = foldername *"linear_solution_IP_convergence.csv"
CSV.write(filename,df)
################################################################################
