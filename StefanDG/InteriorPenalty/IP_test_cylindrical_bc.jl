using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("interior_penalty.jl")
include("../cylinder-analytical-solution.jl")
include("../useful_routines.jl")

function assemble_two_phase_source!(
    systemrhs,
    rhsfunc1,
    rhsfunc2,
    basis,
    cellquads,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            InteriorPenalty.assemble_cell_source!(
                systemrhs,
                rhsfunc1,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            InteriorPenalty.assemble_cell_source!(
                systemrhs,
                rhsfunc2,
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


nelmts = 33
solverorder = 1
levelsetorder = 2
k1 = k2 = 1.0
penaltyfactor = 1e3
center = [0.5, 0.5]
innerradius = 0.4
outerradius = 1.0
q1 = 1.0
q2 = 1.0
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
    x -> [0.0],
    x -> [analyticalsolution(x, center)],
    solverbasis,
    cellquads,
    facequads,
    penalty,
    mergedmesh,
)
assemble_two_phase_source!(
    sysrhs,
    x -> [q1],
    x -> [q2],
    solverbasis,
    cellquads,
    mergedmesh,
)
# ################################################################################
matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 1)
sol = matrix \ rhs
# ################################################################################
#
# ################################################################################
err = mesh_L2_error(
    sol',
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
# ################################################################################

normalizederr = err[1]/den[1]
