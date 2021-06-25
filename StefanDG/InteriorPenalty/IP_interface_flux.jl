using PyPlot
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

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

function plot_interface_flux_error(
    angularposition,
    fluxerror1,
    fluxerror2;
    filename = "",
    ylim = (),
)
    fig, ax = PyPlot.subplots(figsize=(9,3))
    ax.plot(angularposition, fluxerror1, label = "core")
    ax.plot(angularposition, fluxerror2, label = "rim")
    ax.grid()
    ax.legend()
    if length(ylim) > 0
        ax.set_ylim(ylim)
    end
    if length(filename) > 0
        fig.savefig(filename)
        return fig
    else
        return fig
    end
end

function plot_interface_flux_error(
    angularposition,
    fluxerror;
    filename = "",
    ylim = (),
)
    fig, ax = PyPlot.subplots(figsize=(9,3))
    ax.plot(angularposition, fluxerror)
    ax.grid()
    if length(ylim) > 0
        ax.set_ylim(ylim)
    end
    if length(filename) > 0
        fig.savefig(filename)
        return fig
    else
        return fig
    end
end


nelmts = 17
solverorder = 3
numqp = required_quadrature_order(solverorder) + 2
levelsetorder = 2
k1 = 1.0
k2 = 2.0
q1 = 1.0
q2 = 0.0
penaltyfactor = 1e2
center = [0.5, 0.5]
innerradius = 0.4
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
cgmesh = CutCellDG.CGMesh(
    [0.0, 0.0],
    [1.0, 1.0],
    [nelmts, nelmts],
    number_of_basis_functions(levelsetbasis),
)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis)
elementsize = CutCellDG.element_size(dgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)

penalty = penaltyfactor / minelmtsize
tol = minelmtsize^(solverorder + 1)
boundingradius = 1.5 * maxelmtsize

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
grads1 = InteriorPenalty.gradient_at_reference_points(
    sol,
    solverbasis,
    closestrefpoints1,
    closestcellids,
    +1,
    mergedmesh,
)
normalflux1 = k1 * vec(mapslices(sum, grads1 .* normals, dims = 1))
################################################################################

################################################################################
grads2 = InteriorPenalty.gradient_at_reference_points(
    sol,
    solverbasis,
    closestrefpoints2,
    closestcellids,
    -1,
    mergedmesh,
)
normalflux2 = k2 * vec(mapslices(sum, grads2 .* normals, dims = 1))
################################################################################

################################################################################
meanflux = 0.5 * (normalflux1 + normalflux2)
################################################################################

################################################################################
exactflux = -0.5 * q1 * innerradius
fluxerror1 = abs.(normalflux1 .- exactflux) ./ abs(exactflux)
fluxerror2 = abs.(normalflux2 .- exactflux) ./ abs(exactflux)
meanfluxerror = abs.(meanflux .- exactflux) ./ abs(exactflux)
################################################################################

# foldername = "InteriorPenalty\\cylindrical-bc-flux\\"
# filename =
#     foldername *
#     "solverorder-" *
#     string(solverorder) *
#     "-nelmts-" *
#     string(nelmts) *
#     ".png"
# plot_interface_flux_error(
#     angularposition,
#     fluxerror1,
#     fluxerror2,
#     filename = filename,
#     ylim = (0, 0.05),
# )

# filename =
#     foldername *
#     "solverorder-" *
#     string(solverorder) *
#     "-nelmts-" *
#     string(nelmts) *
#     "-mean-flux.png"
plot_interface_flux_error(
    angularposition,
    meanfluxerror,
    # filename = filename,
    ylim = (0, 0.05),
)
