module FiniteDifference
using SparseArrays

struct SystemMatrix
    rows::Any
    cols::Any
    vals::Any
    function SystemMatrix(rows, cols, vals)
        @assert length(rows) == length(cols) == length(vals)
        new(rows, cols, vals)
    end
end

function SystemMatrix()
    rows = Int[]
    cols = Int[]
    vals = zeros(0)
    SystemMatrix(rows, cols, vals)
end

function Base.show(io::IO, sysmatrix::SystemMatrix)
    numvals = length(sysmatrix.rows)
    str = "SystemMatrix with $numvals entries"
    print(io, str)
end

function assemble!(matrix, rows, cols, vals)
    @assert length(rows) == length(cols) == length(vals)
    append!(matrix.rows, rows)
    append!(matrix.cols, cols)
    append!(matrix.vals, vals)
end

function assemble_central_second_derivative!(matrix, nodeid, stepsize)
    rows = repeat([nodeid], 3)
    cols = [nodeid - 1, nodeid, nodeid + 1]
    vals = [1.0, -2.0, 1.0] / stepsize^2
    assemble!(matrix, rows, cols, vals)
end

function assemble_backward_second_derivative!(matrix, nodeid, stepsize)
    rows = repeat([nodeid], 4)
    cols = [nodeid - 3, nodeid - 2, nodeid - 1, nodeid]
    vals = [-1.0, 4.0, -5.0, 2.0] / stepsize^2
    assemble!(matrix, rows, cols, vals)
end

function assemble_forward_second_derivative!(matrix, nodeid, stepsize)
    rows = repeat([nodeid], 4)
    cols = [nodeid, nodeid + 1, nodeid + 2, nodeid + 3]
    vals = [2.0, -5.0, 4.0, -1.0] / stepsize^2
    assemble!(matrix, rows, cols, vals)
end

function away_from_interface(phaseid, idx)
    return phaseid[idx-1] == phaseid[idx] == phaseid[idx+1]
end

function interface_is_ahead(phaseid, idx)
    checkidx = phaseid[idx]
    return phaseid[idx+1] != checkidx &&
           (phaseid[idx-3] == phaseid[idx-2] == phaseid[idx-1] == checkidx)
end

function interface_is_behind(phaseid, idx)
    checkidx = phaseid[idx]
    return phaseid[idx-1] != checkidx &&
           (phaseid[idx+3] == phaseid[idx+2] == phaseid[idx+1] == checkidx)
end

function assemble_laplace_operator!(matrix, phaseid, stepsize)
    numnodes = length(phaseid)
    for idx = 2:numnodes-1
        if away_from_interface(phaseid, idx)
            assemble_central_second_derivative!(matrix, idx, stepsize)
        end
        if phaseid[idx] == 1
            if interface_is_ahead(phaseid, idx)
                assemble_backward_second_derivative!(matrix, idx, stepsize)
            elseif interface_is_behind(phaseid, idx)
                assemble_forward_second_derivative!(matrix, idx, stepsize)
            end
        end
    end
end

function assemble_dirichlet_boundary_condition!(matrix, numnodes)
    assemble!(matrix, [1], [1], [1.0])
    assemble!(matrix, [numnodes], [numnodes], [1.0])
end

function phase_id(phi)
    numnodes = length(phi)
    phaseid = zeros(Int, numnodes)
    for i = 1:numnodes
        if phi[i] < 0
            phaseid[i] = 1
        else
            phaseid[i] = 2
        end
    end
    return phaseid
end

function sparse_operator(sysmatrix, ndofs)
    return dropzeros!(
        sparse(sysmatrix.rows, sysmatrix.cols, sysmatrix.vals, ndofs, ndofs),
    )
end

function assemble_interface_continuity!(matrix, r, s, nodeid)
    rows = repeat([nodeid], 6)
    cols = [nodeid - 2, nodeid - 1, nodeid, nodeid + 1, nodeid + 2, nodeid + 3]
    vals = [r, -4.0r, (1.0 + 3.0r), -(1.0 + 3.0s), 4.0s, -s]
    assemble!(matrix, rows, cols, vals)
end

end
