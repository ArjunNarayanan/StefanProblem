using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
include("../useful_routines.jl")


solverorder = 1
levelsetorder = 1
nelmts = 1
penaltyfactor = 1.0
beta = [1.0, 1.0]
k1 = k2 = 1.0
interfaceangle = 0.0

center = [1.5, 1.0]
radius = 1.0
distancefunction(x) = circle_distance_function(x, center, radius)


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

jacobian = CutCellDG.jacobian(mergedmesh)
detjac = CutCellDG.determinant_jacobian(mergedmesh)
facedetjac = CutCellDG.face_determinant_jacobian(mergedmesh)

D = LocalDG.divergence_operator(solverbasis, cellquads[-1, 1], jacobian, detjac)
facenormals = CutCellDG.reference_face_normals()
R = sum([LocalDG.boundary_face_scalar_flux(
    x -> 1.0,
    solverbasis,
    facequads[-1, faceid, 1],
    facenormals[faceid],
    CutCellDG.cell_map(mergedmesh, -1, 1),
    facedetjac[faceid]
) for faceid in 1:4])
interfacenormals = CutCellDG.interface_normals(interfacequads,1)
