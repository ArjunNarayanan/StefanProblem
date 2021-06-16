using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("interior_penalty.jl")
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
            InteriorPenalty.assemble_cell_source!(
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

function measure_error(
    nelmts,
    solverorder,
    levelsetorder,
    distancefunction,
    negativesource,
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
        x -> [0.0],
        x -> [exactsolution(x)],
        solverbasis,
        cellquads,
        facequads,
        penalty,
        mergedmesh,
    )
    assemble_source_on_negative_mesh!(
        sysrhs,
        negativesource,
        solverbasis,
        cellquads,
        mergedmesh,
    )
    ################################################################################
    matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
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
powers = [2, 3, 4, 5, 6]
nelmts = 2 .^ powers .+ 1
solverorder = 1
levelsetorder = 2
k1 = 1.0
k2 = 2.0
penaltyfactor = 1e3
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

err1 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> [q],
        x -> analyticalsolution(x, center),
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate1 = convergence_rate(dx, err1)
################################################################################


################################################################################
powers = [2, 3, 4, 5, 6]
nelmts = 2 .^ powers .+ 1
solverorder = 2
levelsetorder = 2
k1 = 1.0
k2 = 2.0
penaltyfactor = 1e3
center = [1.0, 1.0]
innerradius = 0.5
outerradius = 1.5
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

err2 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> [q],
        x -> analyticalsolution(x, center),
        k1,
        k2,
        penaltyfactor,
    ) for ne in nelmts
]


dx = 1.0 ./ nelmts
rate2 = convergence_rate(dx, err2)
################################################################################
