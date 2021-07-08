function assemble_boundary_rhs!(systemrhs,TL,TR,basis,penalty,mesh)
    let
        cellid = 1
        qp = -1.0

        rhs = penalty*TL*basis(qp)
        nodeids = nodal_connectivity(mesh,cellid)
        CutCellDG.assemble_cell_rhs!(systemrhs,nodeids,1,rhs)
    end

    let
        cellid = number_of_elements(mesh)
        qp = 1.0

        rhs = penalty*TR*basis(qp)
        nodeids = nodal_connectivity(mesh,cellid)
        CutCellDG.assemble_cell_rhs!(systemrhs,nodeids,1,rhs)
    end
end
