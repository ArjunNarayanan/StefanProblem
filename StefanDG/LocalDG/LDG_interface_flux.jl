using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
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
            LocalDG.assemble_cell_source!(
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
            LocalDG.assemble_cell_source!(
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

nelmts = 17
solverorder = 2
numqp = required_quadrature_order(solverorder) + 2
levelsetorder = 2
k1 = 1.0
k2 = 2.0
q1 = 1.0
q2 = 2.0
penaltyfactor = 1e1
beta = 0.5 * [1.0, 1.0]

center = [0.5, 0.5]
innerradius = 0.3
outerradius = 1.0
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

solverbasis = LagrangeTensorProductBasis(2, solverorder)
levelsetbasis = LagrangeTensorProductBasis(2, levelsetorder)

dgmesh = CutCellDG.DGMesh(
    [0.0, 0.0],
    [1.0, 1.0],
    [nelmts, nelmts],
    interpolation_points(solverbasis),
)
dim, numpts = size(interpolation_points(levelsetbasis))
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [1.0, 1.0], [nelmts, nelmts], numpts)

levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis)


elementsize = CutCellDG.element_size(dgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
tol = minelmtsize^(solverorder + 1)
boundingradius = 1.5 * maxelmtsize


cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh = CutCellDG.MergedMesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads,
    tinyratio = 0.3,
)

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
assemble_two_phase_source!(
    sysrhs,
    x -> q1,
    x -> q2,
    solverbasis,
    cellquads,
    mergedmesh,
)
matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 3)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 3)
sol = reshape(matrix \ rhs, 3, :)

T = sol[1, :]
G = sol[2:3, :]

numquerypoints = 1000
querypoints =
    center .+
    innerradius * hcat(
        [
            [cos(t), sin(t)] for
            t in range(0, stop = 2pi, length = numquerypoints)
        ]...,
    )

cutmesh = CutCellDG.background_mesh(mergedmesh)
refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)
closestpoints, closestcellids = CutCellDG.closest_points_on_zero_levelset(
    querypoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)

angularposition = angular_position(closestpoints .- center)
sortidx = sortperm(angularposition)

angularposition = angularposition[sortidx]
closestpoints = closestpoints[:, sortidx]
closestcellids = closestcellids[sortidx]

normals =
    -1.0CutCellDG.collect_normals_at_spatial_points(
        closestpoints,
        closestcellids,
        levelset,
    )

################################################################################
closestrefpoints1 = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    +1,
    mergedmesh,
)
closestrefpoints2 = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    -1,
    mergedmesh,
)
################################################################################

################################################################################
grads1 = CutCellDG.interpolate_at_reference_points(
    G,
    2,
    solverbasis,
    closestrefpoints1,
    closestcellids,
    +1,
    mergedmesh,
)
normalflux1 = k1 * vec(mapslices(sum, grads1 .* normals, dims = 1))
################################################################################

################################################################################
grads2 = CutCellDG.interpolate_at_reference_points(
    G,
    2,
    solverbasis,
    closestrefpoints2,
    closestcellids,
    -1,
    mergedmesh,
)
normalflux2 = k2 * vec(mapslices(sum, grads2 .* normals, dims = 1))
################################################################################

################################################################################
exactflux = -0.5*q1*innerradius
fluxerror1 = abs.(normalflux1 .- exactflux) ./ abs(exactflux)
fluxerror2 = abs.(normalflux2 .- exactflux) ./ abs(exactflux)
################################################################################


maxerridx = argmax(fluxerror1)
maxerrcellid = closestcellids[maxerridx]
cellsign = CutCellDG.cell_sign(mergedmesh,maxerrcellid)
CutCellDG.load_coefficients!(levelset,maxerrcellid)
interpolater = CutCellDG.interpolater(levelset)

using Plots
xrange = -1:1e-1:1
Plots.contour(xrange,xrange,(x,y)->interpolater([x,y]),levels=[0.0])

# using PyPlot
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, fluxerror1, label = "error 1")
# ax.plot(angularposition, fluxerror2, label = "error 2")
# ax.grid()
# ax.legend()
# fig