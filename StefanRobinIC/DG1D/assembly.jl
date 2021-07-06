function sparse_operator(sysmatrix,mesh,dofspernode)
    numnodes = number_of_nodes(mesh)
    totaldofs = numnodes*dofspernode
    return CutCellDG.sparse_operator(sysmatrix,totaldofs)
end

function rhs_vector(sysrhs,mesh,dofspernode)
    numnodes = number_of_nodes(mesh)
    totaldofs = numnodes*dofspernode
    return CutCellDG.rhs_vector(sysrhs,totaldofs)
end
