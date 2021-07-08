function mass_operator(basis, qp1, qp2)
    V1 = basis(qp1)
    V2 = basis(qp2)
    return V1 * V2'
end

function assemble_edge_penalty_operator!(
    sysmatrix,
    basis,
    qp1,
    qp2,
    penalty,
    nodeids1,
    nodeids2,
)

    M11 = vec(penalty * mass_operator(basis, qp1, qp1))
    M12 = -vec(penalty * mass_operator(basis, qp1, qp2))
    M21 = -vec(penalty * mass_operator(basis, qp2, qp1))
    M22 = vec(penalty * mass_operator(basis, qp2, qp2))

    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids1,
        1,
        M11,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        1,
        M12,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        1,
        M21,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        M22,
    )
end

function assemble_penalty_operator!(sysmatrix, basis, penalty, mesh)
    ncells = number_of_elements(mesh)
    for cellid = 1:(ncells-1)
        nbrcellid = cellid + 1

        nodeids1 = nodal_connectivity(mesh, cellid)
        nodeids2 = nodal_connectivity(mesh, nbrcellid)

        assemble_edge_penalty_operator!(
            sysmatrix,
            basis,
            +1.0,
            -1.0,
            penalty,
            nodeids1,
            nodeids2,
        )
    end
end

function assemble_boundary_penalty_operator!(sysmatrix, basis, penalty, mesh)
    let
        cellid = 1
        qp = -1.0

        M = vec(penalty * mass_operator(basis, qp, qp))
        nodeids = nodal_connectivity(mesh,cellid)
        CutCellDG.assemble_couple_cell_matrix!(
            sysmatrix,
            nodeids,
            nodeids,
            1,
            M,
        )
    end

    let
        cellid = number_of_elements(mesh)
        qp = 1.0

        M = vec(penalty * mass_operator(basis, qp, qp))
        nodeids = nodal_connectivity(mesh,cellid)
        CutCellDG.assemble_couple_cell_matrix!(
            sysmatrix,
            nodeids,
            nodeids,
            1,
            M,
        )
    end
end
