using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
include("../useful_routines.jl")


solverorder = 1
levelsetorder = 1
nelmts = 5
penaltyfactor = 1.0
beta = [1.0, 1.0]
k1 = k2 = 1.0

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
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, [1.0, 0.0], [0.5, 0.0]),
    cgmesh,
    levelsetbasis,
)
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

LocalDG.assemble_divergence_operator!(
    sysmatrix,
    solverbasis,
    cellquads,
    mergedmesh,
)
LocalDG.assemble_mass_operator!(sysmatrix, solverbasis, cellquads, mergedmesh)
LocalDG.assemble_gradient_operator!(
    sysmatrix,
    solverbasis,
    cellquads,
    k1,
    k2,
    mergedmesh,
)

LocalDG.assemble_interelement_scalar_flux_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    beta,
    mergedmesh,
)
LocalDG.assemble_interelement_vector_flux_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    k1,
    k2,
    penaltyfactor,
    beta,
    mergedmesh,
)

LocalDG.assemble_interface_scalar_flux_operator!(
    sysmatrix,
    solverbasis,
    interfacequads,
    beta,
    mergedmesh,
)
LocalDG.assemble_interface_vector_flux_operator!(
    sysmatrix,
    solverbasis,
    interfacequads,
    k1,
    k2,
    penaltyfactor,
    beta,
    mergedmesh,
)

matrix = CutCellDG.sparse_operator(sysmatrix,mergedmesh,3)
