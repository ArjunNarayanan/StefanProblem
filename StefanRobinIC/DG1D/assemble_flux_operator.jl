function flux_operator(basis, qp1, qp2, jacobian)
    V = basis(qp1)
    G = gradient(basis, qp2) ./ jacobian
    return V * G'
end

function assemble_edge_flux_operator!(
    sysmatrix,
    basis,
    qp1,
    qp2,
    normal,
    conductivity1,
    conductivity2,
    jacobian1,
    jacobian2,
    nodeids1,
    nodeids2,
)

    M11 = normal * conductivity1 * flux_operator(basis, qp1, qp1, jacobian1)
    M12 = normal * conductivity2 * flux_operator(basis, qp1, qp2, jacobian2)
    M21 = normal * conductivity1 * flux_operator(basis, qp2, qp1, jacobian1)
    M22 = normal * conductivity2 * flux_operator(basis, qp2, qp2, jacobian2)

    vM11 = -0.5vec(M11+M11')
    vM12 = -0.5vec(M12-M21')
    vM21 = -0.5vec(-M21+M12')
    vM22 = -0.5vec(-M22-M22')

    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids1,
        1,
        vM11,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        1,
        vM21,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        1,
        vM12,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        vM22,
    )
end

function assemble_cell_edge_flux_operator!(
    sysmatrix,
    basis,
    normal,
    conductivity1,
    conductivity2,
    mesh,
    cellid1,
    cellid2,
)

    jacobian1 = CutCellDG.jacobian(cell_map(mesh, cellid1))
    jacobian2 = CutCellDG.jacobian(cell_map(mesh, cellid2))

    nodeids1 = nodal_connectivity(mesh, cellid1)
    nodeids2 = nodal_connectivity(mesh, cellid2)

    assemble_edge_flux_operator!(
        sysmatrix,
        basis,
        +1.0,
        -1.0,
        normal,
        conductivity1,
        conductivity2,
        jacobian1,
        jacobian2,
        nodeids1,
        nodeids2,
    )
end

function assemble_flux_operator!(
    sysmatrix,
    basis,
    conductivity1,
    conductivity2,
    mesh,
)

    ncells = number_of_elements(mesh)

    for cellid = 1:(ncells-1)
        nbrcellid = cellid + 1

        k1 = element_label(mesh, cellid) == 1 ? conductivity1 : conductivity2
        k2 = element_label(mesh, nbrcellid) == 1 ? conductivity1 : conductivity2

        assemble_cell_edge_flux_operator!(
            sysmatrix,
            basis,
            1.0,
            k1,
            k2,
            mesh,
            cellid,
            nbrcellid,
        )
    end
end

function assemble_boundary_flux_operator!(
    sysmatrix,
    basis,
    conductivity1,
    conductivity2,
    mesh,
)

    let
        cellid = 1
        qp = -1.0
        normal = -1.0

        k = element_label(mesh, cellid) == 1 ? conductivity1 : conductivity2
        jacobian = CutCellDG.jacobian(cell_map(mesh, cellid))

        M11 = normal*k*flux_operator(basis,qp,qp,jacobian)
        M = -1.0*vec(M11+M11')
        # M = -1.0*vec(M11)

        nodeids = nodal_connectivity(mesh, cellid)
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
        normal = 1.0

        k = element_label(mesh, cellid) == 1 ? conductivity1 : conductivity2
        jacobian = CutCellDG.jacobian(cell_map(mesh, cellid))

        M11 = normal*k*flux_operator(basis,qp,qp,jacobian)
        M = -1.0*vec(M11+M11')
        # M = -1.0*vec(M11)

        nodeids = nodal_connectivity(mesh, cellid)
        CutCellDG.assemble_couple_cell_matrix!(
            sysmatrix,
            nodeids,
            nodeids,
            1,
            M,
        )
    end
end
