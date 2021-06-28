using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("../local_DG.jl")
include("../../useful_routines.jl")

function exact_solution(v)
    x, y = v
    return 3x + 4y
end

function source_term(v, k)
    return 0.0
end

function exact_gradient(v)
    return [3.,4.]
end


function measure_error(
    nelmts,
    solverorder,
    numqp,
    levelsetorder,
    distancefunction,
    sourceterm,
    exactsolution,
    exactgradient,
    k1,
    k2,
    interiorpenalty,
    interfacepenalty,
    negboundarypenalty,
    posboundarypenalty,
    V0,
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
        interiorpenalty,
        interfacepenalty,
        negboundarypenalty,
        posboundarypenalty,
        V0,
        mergedmesh,
    )

    LocalDG.assemble_LDG_rhs!(
        sysrhs,
        sourceterm,
        exactsolution,
        solverbasis,
        cellquads,
        facequads,
        negboundarypenalty,
        posboundarypenalty,
        V0,
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
    # Gnorm = integral_norm_on_mesh(exactgradient, cellquads, mergedmesh, 2)

    return errT[1] / Tnorm[1], errG # ./ Gnorm
end




################################################################################
powers = [1, 2, 3, 4, 5]
nelmts = 2 .^ powers .+ 1
solverorder = 1
numqp = required_quadrature_order(solverorder)
levelsetorder = 1
k1 = k2 = 1.0
interiorpenalty = 0.0
interfacepenalty = 0.0
negboundarypenalty = 0.0
posboundarypenalty = 1.0
theta = 45
V0 = [cosd(theta), sind(theta)]

interfacepoint = [0.8,0.]
interfaceangle = 60.0
interfacenormal = [cosd(interfaceangle),sind(interfaceangle)]
distancefunction(x) = plane_distance_function(x,interfacenormal,interfacepoint)

LDGerr1 = [
    measure_error(
        ne,
        solverorder,
        numqp,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        exact_gradient,
        k1,
        k2,
        interiorpenalty,
        interfacepenalty,
        negboundarypenalty,
        posboundarypenalty,
        V0,
    ) for ne in nelmts
]

LDGerr1T = [er[1] for er in LDGerr1]
LDGerr1G1 = [er[2][1] for er in LDGerr1]
LDGerr1G2 = [er[2][2] for er in LDGerr1]
LDGerr1 = hcat(LDGerr1T,LDGerr1G1,LDGerr1G2)

dx = 1.0 ./ nelmts

Trate1 = convergence_rate(dx, LDGerr1T)
G1rate1 = convergence_rate(dx, LDGerr1G1)
G2rate1 = convergence_rate(dx, LDGerr1G2)
################################################################################



################################################################################
solverorder = 2
numqp = required_quadrature_order(solverorder)+2

LDGerr2 = [
    measure_error(
        ne,
        solverorder,
        numqp,
        levelsetorder,
        distancefunction,
        x -> source_term(x, k1),
        exact_solution,
        exact_gradient,
        k1,
        k2,
        interiorpenalty,
        interfacepenalty,
        negboundarypenalty,
        posboundarypenalty,
        V0,
    ) for ne in nelmts
]

LDGerr2T = [er[1] for er in LDGerr2]
LDGerr2G1 = [er[2][1] for er in LDGerr2]
LDGerr2G2 = [er[2][2] for er in LDGerr2]
LDGerr2 = hcat(LDGerr2T,LDGerr2G1,LDGerr2G2)

dx = 1.0 ./ nelmts
Trate2 = convergence_rate(dx, LDGerr2T)
G1rate2 = convergence_rate(dx, LDGerr2G1)
G2rate2 = convergence_rate(dx, LDGerr2G2)
################################################################################


################################################################################
using DataFrames, CSV
df = DataFrame(NElmts = nelmts,
    errTLinear = LDGerr1T,
    errTQuadratic = LDGerr2T)

foldername = "LocalDG\\plane_interface_tests\\"
filename = foldername *"linear_solution_LDG_convergence.csv"
CSV.write(filename,df)
################################################################################
