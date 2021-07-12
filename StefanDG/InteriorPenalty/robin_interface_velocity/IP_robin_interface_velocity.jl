using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../interior_penalty.jl")
include("../../robin-interface-cylindrical-solution.jl")
include("../../useful_routines.jl")
RAS = RobinAnalyticalSolution

function (solver::RobinAnalyticalSolution.CylindricalSolver)(x, center)
    dx = x - center
    r = sqrt(dx' * dx)
    return RobinAnalyticalSolution.analytical_solution(r, solver)
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

function plot_interface_error(
    angularposition,
    fluxerror1,
    fluxerror2;
    filename = "",
    ylim = (),
)
    fig, ax = PyPlot.subplots(figsize = (9, 3))
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


nelmts = 17
solverorder = 3
levelsetorder = 2
numqp = required_quadrature_order(solverorder) + 2
k1 = 1.0
k2 = 2.0
q1 = 2.0
q2 = 1.0
lambda = 1.0
Tm = 0.5
penaltyfactor = 1e2
center = [0.5, 0.5]
innerradius = 0.4
outerradius = 1.0
Tw = 1.0
distancefunction(x) = circle_distance_function(x, center, innerradius)
analyticalsolution = RobinAnalyticalSolution.CylindricalSolver(
    q1,
    k1,
    q2,
    k2,
    lambda,
    innerradius,
    outerradius,
    Tw,
    Tm,
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
InteriorPenalty.assemble_robin_operator!(
    sysmatrix,
    solverbasis,
    interfacequads,
    lambda,
    mergedmesh,
)
InteriorPenalty.assemble_boundary_source!(
    sysrhs,
    x -> analyticalsolution(x, center),
    solverbasis,
    facequads,
    k1,
    k2,
    penalty,
    mergedmesh,
)
InteriorPenalty.assemble_two_phase_source!(
    sysrhs,
    x -> q1,
    x -> q2,
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

matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 1)
nodalsolution = matrix \ rhs

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
vals1 = InteriorPenalty.interpolate_at_reference_points(
    nodalsolution,
    solverbasis,
    closestrefpoints1,
    closestcellids,
    +1,
    mergedmesh,
)
velocity1 = -lambda * (Tm .- vals1)
################################################################################

################################################################################
vals2 = InteriorPenalty.interpolate_at_reference_points(
    nodalsolution,
    solverbasis,
    closestrefpoints2,
    closestcellids,
    -1,
    mergedmesh,
)
velocity2 = -lambda * (Tm .- vals2)
################################################################################

################################################################################
meanvelocity = 0.5*(velocity1 + velocity2)
################################################################################

################################################################################
exactinterfacetemp = RAS.analytical_solution(innerradius, analyticalsolution)
exactvelocity = -lambda * (Tm - exactinterfacetemp)
################################################################################

################################################################################
error1 = abs.(velocity1 .- exactvelocity) ./ abs(exactvelocity)
error2 = abs.(velocity2 .- exactvelocity) ./ abs(exactvelocity)
meanerror = abs.(meanvelocity .- exactvelocity) ./ abs(exactvelocity)
################################################################################

# using Plots
# plot(meanerror)

plot_interface_error(angularposition, error1, error2)
