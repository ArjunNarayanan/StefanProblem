function assemble_robin_edge_operator!(
    sysmatrix,
    basis,
    qp1,
    qp2,
    lambda,
    nodeids1,
    nodeids2,
)

    M11 = 0.25 * lambda * vec(mass_operator(basis, qp1, qp1))
    M12 = 0.25 * lambda * vec(mass_operator(basis, qp1, qp2))
    M21 = 0.25 * lambda * vec(mass_operator(basis, qp2, qp1))
    M22 = 0.25 * lambda * vec(mass_operator(basis, qp2, qp2))

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

function assemble_interface_robin_operator!(sysmatrix, basis, lambda, mesh)
    ncells = number_of_elements(mesh)
    for cellid = 1:(ncells-1)
        nbrcellid = cellid + 1

        l1 = element_label(mesh, cellid)
        l2 = element_label(mesh, nbrcellid)

        if l1 != l2
            nodeids1 = nodal_connectivity(mesh, cellid)
            nodeids2 = nodal_connectivity(mesh, nbrcellid)

            assemble_robin_edge_operator!(
                sysmatrix,
                basis,
                1.0,
                -1.0,
                lambda,
                nodeids1,
                nodeids2,
            )
        end
    end
end
