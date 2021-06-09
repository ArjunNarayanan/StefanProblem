function assemble_cell_source!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads,
    mesh,
    cellsign,
    cellid,
)
    detjac = CutCellDG.determinant_jacobian(mesh)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)
    quad = cellquads[cellsign, cellid]
    rhs = CutCellDG.linear_form(rhsfunc, basis, quad, cellmap, 1, detjac)
    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
end

function assemble_source!(systemrhs, rhsfunc, basis, cellquads, mesh)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                -1,
                cellid,
            )
        end
    end
end

function assemble_boundary_face_source!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    penalty,
    cellmap,
    facedetjac,
    nodeids,
)

    rhs =
        -1.0 *
        penalty *
        CutCellDG.linear_form(rhsfunc, basis, quad, cellmap, 1, facedetjac)
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
end

function assemble_boundary_cell_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    penalty,
    mesh,
    cellsign,
    cellid,
    faceids,
    facedetjac,
)

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0
            assemble_boundary_face_source!(
                systemrhs,
                rhsfunc,
                basis,
                quad,
                penalty,
                cellmap,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    penalty,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_boundary_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                penalty,
                mesh,
                +1,
                cellid,
                faceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                penalty,
                mesh,
                -1,
                cellid,
                faceids,
                facedetjac,
            )
        end
    end
end
