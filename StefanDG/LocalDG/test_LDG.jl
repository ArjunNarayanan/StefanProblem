using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("local_DG.jl")
include("../useful_routines.jl")

function exact_solution(v)
    x, y = v
    return cos(2pi * x) * sin(2pi * y)
end

function exact_gradient(v)
    x, y = v
    gx = -2pi * sin(2pi * x) * sin(2pi * y)
    gy = 2pi * cos(2pi * x) * cos(2pi * y)
    return [gx, gy]
end

function source_term(v, k)
    x, y = v
    return -8pi^2 * k * cos(2pi * x) * sin(2pi * y)
end

solverorder = 1
levelsetorder = 1
nelmts = 17
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
    penaltyfactor,
    beta,
    mergedmesh,
)
LocalDG.assemble_LDG_rhs!(
    sysrhs,
    x -> -source_term(x, k1),
    exact_solution,
    solverbasis,
    cellquads,
    facequads,
    penaltyfactor,
    mergedmesh,
)

matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 3)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 3)

sol = reshape(matrix \ rhs, 3, :)
T = sol[1, :]'
G = sol[2:3, :]

errT = mesh_L2_error(T, exact_solution, solverbasis, cellquads, mergedmesh)
errG = mesh_L2_error(G, exact_gradient, solverbasis, cellquads, mergedmesh)

println("Error T = $errT")
println("Error G = $errG")
