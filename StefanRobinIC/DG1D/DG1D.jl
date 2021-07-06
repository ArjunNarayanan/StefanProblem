module DG1D
using PolynomialBasis
using CutCellDG

include("dgmesh_1d.jl")
include("assemble_bilinear_form.jl")
include("assemble_flux_operator.jl")
include("assemble_penalty_operator.jl")
include("assemble_boundary_conditions.jl")
include("assembly.jl")
# end module
end
