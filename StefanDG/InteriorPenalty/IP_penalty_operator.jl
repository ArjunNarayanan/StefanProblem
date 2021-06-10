function mass_operator(basis, quad1, quad2, scaleareas)
    numqp = length(quad1)
    @assert length(quad2) == numqp
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, nf)
    for idx = 1:length(quad1)
        p1, w1 = quad1[idx]
        p2, w2 = quad2[idx]

        V1 = basis(p1)
        V2 = basis(p2)

        matrix .+= V1 * V2' * scaleareas[idx] * w1
    end
    return matrix
end


function assemble_face_penalty_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    penalty,
    scaleareas,
    nodeids1,
    nodeids2,
)

    M11 = vec(penalty * mass_operator(basis, quad1, quad1, scaleareas))
    M12 = vec(penalty * mass_operator(basis, quad1, quad2, scaleareas))
    M21 = vec(penalty * mass_operator(basis, quad2, quad1, scaleareas))
    M22 = vec(penalty * mass_operator(basis, quad2, quad2, scaleareas))

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
        -M12,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        1,
        -M21,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        M22,
    )
end

function assemble_face_interelement_penalty_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    penalty,
    facedetjac,
    nodeids1,
    nodeids2,
)

    scaleareas = repeat([facedetjac], length(quad1))
    assemble_face_penalty_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        penalty,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_cut_cell_interelement_penalty_operator!(
    sysmatrix,
    basis,
    facequads,
    penalty,
    mesh,
    cellsign,
    cellid,
    faceids,
    nbrfaceids,
    facedetjac,
)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        quad1 = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if cellid < nbrcellid &&
           CutCellDG.solution_cell_id(mesh, cellsign, nbrcellid) !=
           CutCellDG.solution_cell_id(mesh, cellsign, cellid)

            nbrcellsign = CutCellDG.cell_sign(mesh, nbrcellid)

            if nbrcellsign == cellsign || nbrcellsign == 0
                quad2 = facequads[cellsign, nbrfaceids[faceid], nbrcellid]
                nodeids2 =
                    CutCellDG.nodal_connectivity(mesh, cellsign, nbrcellid)
                assemble_face_interelement_penalty_operator!(
                    sysmatrix,
                    basis,
                    quad1,
                    quad2,
                    penalty,
                    facedetjac[faceid],
                    nodeids1,
                    nodeids2,
                )
            end
        end
    end
end

function assemble_interelement_penalty_operator!(
    sysmatrix,
    basis,
    facequads,
    penalty,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    nbrfaceids = [CutCellDG.opposite_face(faceid) for faceid in faceids]

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_interelement_penalty_operator!(
                sysmatrix,
                basis,
                facequads,
                penalty,
                mesh,
                +1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_interelement_penalty_operator!(
                sysmatrix,
                basis,
                facequads,
                penalty,
                mesh,
                -1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
            )
        end
    end
end


function assemble_cell_interface_penalty_operator!(
    sysmatrix,
    basis,
    interfacequads,
    penalty,
    mesh,
    cellid,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_penalty_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        penalty,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_interface_penalty_operator!(
    sysmatrix,
    basis,
    interfacequads,
    penalty,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_interface_penalty_operator!(
                sysmatrix,
                basis,
                interfacequads,
                penalty,
                mesh,
                cellid,
            )
        end
    end
end

function assemble_boundary_face_penalty_operator!(
    sysmatrix,
    basis,
    quad,
    penalty,
    facedetjac,
    nodeids,
)

    numqp = length(quad)
    scaleareas = repeat([facedetjac], numqp)

    M = vec(penalty * mass_operator(basis, quad, quad, scaleareas))

    CutCellDG.assemble_couple_cell_matrix!(sysmatrix, nodeids, nodeids, 1, M)
end

function assemble_boundary_cell_penalty_operator!(
    sysmatrix,
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
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0
            assemble_boundary_face_penalty_operator!(
                sysmatrix,
                basis,
                quad,
                penalty,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_penalty_operator!(
    sysmatrix,
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
            assemble_boundary_cell_penalty_operator!(
                sysmatrix,
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
            assemble_boundary_cell_penalty_operator!(
                sysmatrix,
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
