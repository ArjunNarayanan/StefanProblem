using PolynomialBasis
using ImplicitDomainQuadrature
using CutCellDG
include("DG1D.jl")
include("../useful_routines.jl")
include("../analytical_solution_1d.jl")
AS = AnalyticalSolver


q1, q2 = 1.0, 1.0
k1, k2 = 0.5, 0.5
lambda = 0.0
interfacepoint = 0.5
TL = 0.0
TR = 1.0
Tm = 0.5

exactsolution =
    AS.Analytical1D(q1, k1, q2, k2, lambda, interfacepoint, TL, TR, Tm)

ne = 1
penaltyfactor = 1e2
solverorder = 2
solverbasis = LagrangeTensorProductBasis(1, solverorder)
numqp = required_quadrature_order(solverorder) + 1
quad = tensor_product_quadrature(1, numqp)




refpoints = interpolation_points(solverbasis)
mesh = DG1D.DGMesh1D(0.0, 1.0, interfacepoint, ne, ne, refpoints)
minelmtsize = minimum(DG1D.element_size(mesh))
penalty = penaltyfactor / minelmtsize

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

DG1D.assemble_gradient_operator!(sysmatrix, solverbasis, quad, k1, k2, mesh)
DG1D.assemble_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
DG1D.assemble_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
DG1D.assemble_two_phase_source!(
    sysrhs,
    x -> q1,
    x -> q2,
    solverbasis,
    quad,
    mesh,
)
DG1D.assemble_boundary_flux_operator!(sysmatrix, solverbasis, k1, k2, mesh)
DG1D.assemble_boundary_penalty_operator!(sysmatrix, solverbasis, penalty, mesh)
DG1D.assemble_boundary_rhs!(sysrhs, TL, TR, solverbasis, penalty, mesh)

matrix = DG1D.sparse_operator(sysmatrix, mesh, 1)
rhs = DG1D.rhs_vector(sysrhs, mesh, 1)

solution = matrix \ rhs
nodalcoordinates = vec(DG1D.nodal_coordinates(mesh))
truesolution = exactsolution.(nodalcoordinates)

using Plots
plot(nodalcoordinates,solution)
plot!(nodalcoordinates,truesolution)
