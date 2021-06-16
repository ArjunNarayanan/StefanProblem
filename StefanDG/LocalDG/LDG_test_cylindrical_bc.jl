using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
include("../cylinder-analytical-solution.jl")
include("../useful_routines.jl")

function assemble_source_on_negative_mesh!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == -1 || cellsign == 0
            LocalDG.assemble_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                -1,
                cellid,
            )
        end
    end
end

function (solver::AnalyticalSolution.CylindricalSolver)(x, center)
    dx = x - center
    r = sqrt(dx' * dx)
    return AnalyticalSolution.analytical_solution(r, solver)
end

nelmts = 9
solverorder = 1
numqp = required_quadrature_order(solverorder)
levelsetorder = 2
k1 = k2 = 1.0
penaltyfactor = 1.0
beta = 0.5 * [1.0, 1.0]

center = [0.5, 0.5]
innerradius = 0.4
outerradius = 1.0
q = 1.0
Tw = 1.0

distancefunction(x) = circle_distance_function(x, center, innerradius)
analyticalsolution = AnalyticalSolution.CylindricalSolver(
    q,
    k1,
    k2,
    innerradius,
    outerradius,
    Tw,
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

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh =
    CutCellDG.MergedMesh!(cutmesh, cellquads, facequads, interfacequads)

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

LocalDG.assemble_LDG_linear_system!(
    sysmatrix,
    solverbasis,
    cellquads,
    facequads,
    interfacequads,
    k1,
    k2,
    penaltyfactor,
    beta,
    mergedmesh,
)
LocalDG.assemble_LDG_rhs!(
    sysrhs,
    x -> 0.0,
    x -> analyticalsolution(x, center),
    solverbasis,
    cellquads,
    facequads,
    penaltyfactor,
    mergedmesh,
)
assemble_source_on_negative_mesh!(
    sysrhs,
    x -> q,
    solverbasis,
    cellquads,
    mergedmesh,
)
matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 3)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 3)
sol = reshape(matrix \ rhs, 3, :)

T = sol[1, :]
G = sol[2:3, :]

err = mesh_L2_error(
    T',
    x -> analyticalsolution(x, center),
    solverbasis,
    cellquads,
    mergedmesh,
)
den = integral_norm_on_mesh(
    x -> analyticalsolution(x, center),
    cellquads,
    mergedmesh,
    1,
)

normalizederr = err[1] / den[1]
