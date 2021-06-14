function scalar_flux_operator(basis, quad1, quad2, normals, scaleareas)
    numqp = length(quad1)
    @assert length(quad2) == numqp
    @assert size(normals) == (2, numqp)
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    matrix = zeros(2nf, nf)
    for idx = 1:numqp
        p1, w1 = quad1[idx]
        p2, w2 = quad2[idx]
        @assert w1 ≈ w2

        n = normals[:, idx]
        a = scaleareas[idx]

        VL = CutCellDG.interpolation_matrix(basis(p1), 2)
        VR = basis(p2)

        matrix .+= VL' * n * VR' * a * w1
    end
    return matrix
end

function assemble_face_scalar_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normals,
    beta,
    scaleareas,
    nodeids1,
    nodeids2,
)

    M11 =
        -0.5 *
        vec(scalar_flux_operator(basis, quad1, quad1, normals, scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [2, 3],
        nodeids1,
        [1],
        3,
        M11,
    )

    M12 =
        -0.5 *
        vec(scalar_flux_operator(basis, quad1, quad2, normals, scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [2, 3],
        nodeids2,
        [1],
        3,
        M12,
    )

    bp = normals' * beta
    bm = -bp

    N11 =
        -1.0 * vec(
            scalar_flux_operator(
                basis,
                quad1,
                quad1,
                normals,
                bp .* scaleareas,
            ),
        )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [2, 3],
        nodeids1,
        [1],
        3,
        N11,
    )

    N12 =
        -1.0 * vec(
            scalar_flux_operator(
                basis,
                quad1,
                quad2,
                normals,
                bm .* scaleareas,
            ),
        )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [2, 3],
        nodeids2,
        [1],
        3,
        N12,
    )
end


################################################################################
function assemble_face_interelement_scalar_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normal,
    beta,
    facedetjac,
    nodeids1,
    nodeids2,
)

    numqp = length(quad1)
    normals = repeat(normal, inner = (1, numqp))
    scaleareas = repeat([facedetjac], numqp)

    assemble_face_scalar_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        normals,
        beta,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_cut_cell_interelement_scalar_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    beta,
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

        if nbrcellid != 0 &&
           CutCellDG.solution_cell_id(mesh, cellsign, nbrcellid) !=
           CutCellDG.solution_cell_id(mesh, cellsign, cellid)

            nbrfaceid = nbrfaceids[faceid]
            nbrcellsign = CutCellDG.cell_sign(mesh, nbrcellid)

            if nbrcellsign == cellsign || nbrcellsign == 0
                quad2 = facequads[cellsign, nbrfaceid, nbrcellid]
                nodeids2 =
                    CutCellDG.nodal_connectivity(mesh, cellsign, nbrcellid)

                assemble_face_interelement_scalar_flux_operator!(
                    sysmatrix,
                    basis,
                    quad1,
                    quad2,
                    normals[faceid],
                    beta,
                    facedetjac[faceid],
                    nodeids1,
                    nodeids2,
                )
            end
        end
    end
end

function assemble_interelement_scalar_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    beta,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    nbrfaceids = [CutCellDG.opposite_face(faceid) for faceid in faceids]
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_interelement_scalar_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                beta,
                mesh,
                +1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_interelement_scalar_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                beta,
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
################################################################################


################################################################################
function assemble_cell_interface_scalar_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    beta,
    mesh,
    cellid,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    negnormals = CutCellDG.interface_normals(interfacequads, cellid)
    posnormals = -1.0 * negnormals
    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_scalar_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        posnormals,
        beta,
        scaleareas,
        nodeids1,
        nodeids2,
    )
    assemble_face_scalar_flux_operator!(
        sysmatrix,
        basis,
        quad2,
        quad1,
        negnormals,
        beta,
        scaleareas,
        nodeids2,
        nodeids1,
    )
end

function assemble_interface_scalar_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    beta,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_interface_scalar_flux_operator!(
                sysmatrix,
                basis,
                interfacequads,
                beta,
                mesh,
                cellid,
            )
        end
    end
end
################################################################################
