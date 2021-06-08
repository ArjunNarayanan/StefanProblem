include("finite-difference.jl")
using LinearAlgebra
FD = FiniteDifference

xI = 0.5
numnodes = 9
xrange = range(0,stop=1,length=numnodes)
stepsize = 1.0/(numnodes-1)
phi = xI .- xrange
phaseid = FD.phase_id(phi)

matrix = FD.SystemMatrix()
FD.assemble_laplace_operator!(matrix,phaseid,stepsize)
FD.assemble_dirichlet_boundary_condition!(matrix,numnodes)
K = FD.sparse_operator(matrix,numnodes)
