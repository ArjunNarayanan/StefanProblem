using LinearAlgebra, Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("interior_penalty.jl")
include("../useful_routines.jl")

function exact_solution(v)
    x, y = v
    return 3x + 4y
end

function source_term(v, k)
    return 0.0
end

k1 = k2 = 1.0
solverorder = 1
levelsetorder = 1
nelmts = 3
penaltyfactor = 1e3
numqp = required_quadrature_order(solverorder)
solverbasis = LagrangeTensorProductBasis(2, solverorder)
levelsetbasis = LagrangeTensorProductBasis(2, levelsetorder)

distancefunction(x) = plane_distance_function(x, [1.0, 0.0], [0.5, 0.0])

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

################################################################################
# LINEAR SYSTEM
InteriorPenalty.assemble_gradient_operator!(
    sysmatrix,
    solverbasis,
    cellquads,
    k1,
    k2,
    mergedmesh,
)
################################################################################

################################################################################
# INTERELEMENT CONDITION
InteriorPenalty.assemble_interelement_flux_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    k1,
    k2,
    mergedmesh,
)
InteriorPenalty.assemble_interelement_penalty_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    penalty,
    mergedmesh,
)
# ################################################################################

################################################################################
# INTERFACE CONDITION
InteriorPenalty.assemble_interface_flux_operator!(
    sysmatrix,
    solverbasis,
    interfacequads,
    k1,
    k2,
    mergedmesh,
)
InteriorPenalty.assemble_interface_penalty_operator!(
    sysmatrix,
    solverbasis,
    interfacequads,
    penalty,
    mergedmesh,
)
################################################################################

# ################################################################################
# # BOUNDARY CONDITIONS
InteriorPenalty.assemble_boundary_flux_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    k1,
    k2,
    mergedmesh,
)
InteriorPenalty.assemble_boundary_penalty_operator!(
    sysmatrix,
    solverbasis,
    facequads,
    penalty,
    mergedmesh,
)
InteriorPenalty.assemble_boundary_source!(
    sysrhs,
    x -> exact_solution(x),
    solverbasis,
    facequads,
    k1,
    k2,
    penalty,
    mergedmesh,
)
# ################################################################################
# # SOURCE TERM
InteriorPenalty.assemble_source!(
    sysrhs,
    x -> source_term(x, k1),
    solverbasis,
    cellquads,
    cutmesh,
)
# ################################################################################
#
# ################################################################################
matrix = CutCellDG.sparse_operator(sysmatrix, mergedmesh, 1)
rhs = CutCellDG.rhs_vector(sysrhs, mergedmesh, 1)

sol = matrix \ rhs
################################################################################


################################################################################
err = mesh_L2_error(sol', exact_solution, solverbasis, cellquads, mergedmesh)
@test isapprox(err[1],0.0,atol=1e4eps())
@test issymmetric(matrix)
# println("Error = ", err[1])
################################################################################
