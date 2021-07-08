function assemble_robin_edge_source!(
    sysrhs,
    basis,
    qp1,
    qp2,
    lambda,
    Tm,
    nodeids1,
    nodeids2,
)

    rhs1 = 0.5 * lambda * Tm * basis(qp1)
    rhs2 = 0.5 * lambda * Tm * basis(qp2)

    CutCellDG.assemble_cell_rhs!(sysrhs, nodeids1, 1, rhs1)
    CutCellDG.assemble_cell_rhs!(sysrhs, nodeids2, 1, rhs2)
end

function assemble_interface_robin_source!(sysrhs, basis, lambda, Tm, mesh)
    ncells = number_of_elements(mesh)
    for cellid = 1:(ncells-1)
        nbrcellid = cellid + 1

        l1 = element_label(mesh, cellid)
        l2 = element_label(mesh, nbrcellid)

        if l1 != l2
            nodeids1 = nodal_connectivity(mesh, cellid)
            nodeids2 = nodal_connectivity(mesh, nbrcellid)

            assemble_robin_edge_source!(
                sysrhs,
                basis,
                +1.0,
                -1.0,
                lambda,
                Tm,
                nodeids1,
                nodeids2,
            )
        end
    end
end
