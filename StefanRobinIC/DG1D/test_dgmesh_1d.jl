using PolynomialBasis
using CutCellDG
include("DG1D.jl")

solverorder = 2
ne1 = 2
ne2 = 2
interfacepoint = 0.5
solverbasis = LagrangeTensorProductBasis(1,solverorder)
refpoints = interpolation_points(solverbasis)
numrefpoints = size(refpoints)[2]

dgmesh = DG1D.DGMesh1D(0.,1.,interfacepoint,ne1,ne2,refpoints)
