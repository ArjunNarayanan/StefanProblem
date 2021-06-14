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

function measure_error(
    nelmts,
    solverorder,
    levelsetorder,
    distancefunction,
    negativesource,
    exactsolution,
    exactgradient,
    k1,
    k2,
    penaltyfactor,
    beta,
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

    penalty = penaltyfactor
    # penalty = penaltyfactor / minelmtsize * 0.5 * (k1 + k2)

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
        penalty,
        beta,
        mergedmesh,
    )

    LocalDG.assemble_LDG_rhs!(
        sysrhs,
        sourceterm,
        exactsolution,
        solverbasis,
        cellquads,
        facequads,
        penalty,
        mergedmesh,
    )
    ################################################################################
    matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 3)
    rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 3)
    sol = reshape(matrix \ rhs, 3, :)
    ################################################################################

    ################################################################################
    T = sol[1, :]'
    G = sol[2:3, :]

    errT = mesh_L2_error(T, exactsolution, solverbasis, cellquads, mergedmesh)
    errG = mesh_L2_error(G, exactgradient, solverbasis, cellquads, mergedmesh)
    ################################################################################

    Tnorm = integral_norm_on_mesh(exactsolution, cellquads, mergedmesh, 1)
    Gnorm = integral_norm_on_mesh(exactgradient, cellquads, mergedmesh, 2)

    return errT[1] / Tnorm[1], errG ./ Gnorm
end




################################################################################
powers = [2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1.0
beta = 0.5 * [1.0, 1.0]
distancefunction(x) = plane_distance_function(x, [1.0, 0.0], [0.5, 0.0])

err1 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        exact_gradient,
        k1,
        k2,
        penaltyfactor,
        beta,
    ) for ne in nelmts
]

err1T = [er[1] for er in err1]
err1G1 = [er[2][1] for er in err1]
err1G2 = [er[2][2] for er in err1]

dx = 1.0 ./ nelmts

Trate1 = convergence_rate(dx, err1T)
G1rate1 = convergence_rate(dx, err1G1)
G2rate1 = convergence_rate(dx, err1G2)
################################################################################



################################################################################
powers = [2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 2
levelsetorder = 1
k1 = k2 = 1.0
penaltyfactor = 1.0
beta = 0.5 * [1.0, 1.0]
distancefunction(x) = circle_distance_function(x, [0.5, 0.5], 0.3)

err2 = [
    measure_error(
        ne,
        solverorder,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        exact_gradient,
        k1,
        k2,
        penaltyfactor,
        beta,
    ) for ne in nelmts
]

err2T = [er[1] for er in err2]
err2G1 = [er[2][1] for er in err2]
err2G2 = [er[2][2] for er in err2]

dx = 1.0 ./ nelmts

Trate2 = convergence_rate(dx, err2T)
G1rate2 = convergence_rate(dx, err2G1)
G2rate2 = convergence_rate(dx, err2G2)
################################################################################
