module DG1D
using PolynomialBasis
using CutCellDG

include("dgmesh_1d.jl")
include("assemble_gradient_operator.jl")
include("assemble_flux_operator.jl")
include("assemble_penalty_operator.jl")
include("assemble_interface_robin_operator.jl")
include("assemble_interface_robin_source.jl")
include("assemble_boundary_conditions.jl")
include("assemble_source_term.jl")
include("assembly.jl")
# end module
end
