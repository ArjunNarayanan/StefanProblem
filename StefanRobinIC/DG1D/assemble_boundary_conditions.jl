function assemble_boundary_rhs!(
    systemrhs,
    conductivity1,
    conductivity2,
    TL,
    TR,
    basis,
    penalty,
    mesh,
)
    let
        cellid = 1
        qp = -1.0
        normal = -1.0

        k = element_label(mesh, cellid) == 1 ? conductivity1 : conductivity2
        jacobian = CutCellDG.jacobian(cell_map(mesh, cellid))

        R1 = penalty * TL * basis(qp)
        R2 = -TL * k * normal * gradient(basis, qp) / jacobian

        rhs = R1 + R2

        nodeids = nodal_connectivity(mesh, cellid)
        CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
    end

    let
        cellid = number_of_elements(mesh)
        qp = 1.0
        normal = 1.0

        k = element_label(mesh, cellid) == 1 ? conductivity1 : conductivity2
        jacobian = CutCellDG.jacobian(cell_map(mesh, cellid))

        R1 = penalty * TR * basis(qp)
        R2 = -TR * k * normal * gradient(basis, qp) / jacobian

        rhs = R1 + R2

        nodeids = nodal_connectivity(mesh, cellid)
        CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
    end
end
