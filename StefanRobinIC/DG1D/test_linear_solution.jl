using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("DG1D.jl")
include("../useful_routines.jl")

xL = 0.0
xR = 1.0
interfacepoint = 0.5
k1 = k2 = 1.0
q1 = q2 = 0.0
TL = 0.0
TR = 1.0

ne = 1
ne1 = ne
ne2 = ne
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
DG1D.assemble_two_phase_source!(
    sysrhs,
    x -> q1,
    x -> q2,
    solverbasis,
    quad,
    mesh,
)
DG1D.assemble_boundary_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
DG1D.assemble_boundary_rhs!(sysrhs, k1, k2, TL, TR, solverbasis, penalty, mesh)

matrix = DG1D.sparse_operator(sysmatrix, mesh, 1)
rhs = DG1D.rhs_vector(sysrhs, mesh, 1)

solution = matrix \ rhs

err = uniform_mesh_L2_error(solution',x->x[1],solverbasis,quad,mesh)
den = integral_norm_on_uniform_mesh(x->x[1],quad,mesh,1)

err = err[1]/den[1]

@test isapprox(0.0,err,atol=1e3eps())
@test issymmetric(matrix)
