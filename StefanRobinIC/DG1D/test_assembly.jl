using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("DG1D.jl")
include("../useful_routines.jl")

xL = 0.0
xR = 1.0
interfacepoint = 0.5
k1 = k2 = 1.0
TL = TR = 1.0

ne1 = 1
ne2 = 1
penalty = 1e2
solverorder = 1

solverbasis = LagrangeTensorProductBasis(1, solverorder)
numqp = required_quadrature_order(solverorder) + 1
quad = tensor_product_quadrature(1, numqp)
refpoints = interpolation_points(solverbasis)
mesh = DG1D.DGMesh1D(xL, xR, interfacepoint, ne1, ne2, refpoints)

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

DG1D.assemble_gradient_operator!(sysmatrix, solverbasis, quad, k1, k2, mesh)
DG1D.assemble_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
DG1D.assemble_boundary_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
DG1D.assemble_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
DG1D.assemble_boundary_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
DG1D.assemble_boundary_rhs!(sysrhs, TL, TR, solverbasis, penalty, mesh)

matrix = DG1D.sparse_operator(sysmatrix,mesh,1)
rhs = DG1D.rhs_vector(sysrhs,mesh,1)

solution = matrix\rhs
