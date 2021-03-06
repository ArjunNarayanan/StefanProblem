using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
include("../useful_routines.jl")


solverorder = 2
levelsetorder = 1
nelmts = 5
interiorpenalty = 0.0
interfacepenalty = 0.0
negboundarypenalty = 0.0
posboundarypenalty = 1.0
V0 = [1.0, 1.0]
k1 = k2 = 1.0

distancefunction(x) = ones(size(x)[2])


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
    x -> 0.0,
    x -> 1.0,
    solverbasis,
    cellquads,
    facequads,
    negboundarypenalty,
    posboundarypenalty,
    V0,
    mergedmesh,
)

matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 3)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 3)

sol = reshape(matrix \ rhs, 3, :)
T = sol[1, :]'
G = sol[2:3, :]

errT = mesh_L2_error(T, x -> 1.0, solverbasis, cellquads, mergedmesh)
errG = mesh_L2_error(G, x -> [0.0, 0.0], solverbasis, cellquads, mergedmesh)
