function flux_operator(
    basis,
    quad1,
    quad2,
    normals,
    conductivity,
    jacobian,
    scaleareas,
)

    numqp = length(quad1)
    @assert length(quad2) == numqp
    dim, numnormals = size(normals)
    @assert numnormals == numqp
    @assert length(scaleareas) == numqp


    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, nf)
    for idx = 1:length(quad1)
        p1, w1 = quad1[idx]
        n = normals[:, idx]
        p2, w2 = quad2[idx]
        @assert w1 ≈ w2

        G = CutCellDG.transform_gradient(gradient(basis, p1), jacobian)
        V = basis(p2)

        matrix .+= conductivity * G * n * V' * scaleareas[idx] * w1
    end

    return matrix
end

function assemble_face_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normals,
    conductivity1,
    conductivity2,
    jacobian,
    scaleareas,
    nodeids1,
    nodeids2,
)

    M11 = -1.0flux_operator(
        basis,
        quad1,
        quad1,
        normals,
        conductivity1,
        jacobian,
        scaleareas,
    )
    M12 = -1.0flux_operator(
        basis,
        quad1,
        quad2,
        normals,
        conductivity1,
        jacobian,
        scaleareas,
    )
    M21 = -1.0flux_operator(
        basis,
        quad2,
        quad1,
        normals,
        conductivity2,
        jacobian,
        scaleareas,
    )
    M22 = -1.0flux_operator(
        basis,
        quad2,
        quad2,
        normals,
        conductivity2,
        jacobian,
        scaleareas,
    )


    vM11 = 0.5vec(M11)
    vM12 = 0.5vec(M12)
    vM21 = 0.5vec(M21)
    vM22 = 0.5vec(M22)

    vM11T = 0.5vec(transpose(M11))
    vM12T = 0.5vec(transpose(M12))
    vM21T = 0.5vec(transpose(M21))
    vM22T = 0.5vec(transpose(M22))

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
        -vM12,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        -vM22,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids1,
        1,
        vM11T,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        1,
        vM21T,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        1,
        -vM12T,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        -vM22T,
    )

end


################################################################################
function assemble_face_interelement_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normal,
    conductivity,
    jacobian,
    facedetjac,
    nodeids1,
    nodeids2,
)

    numqp = length(quad1)
    normals = repeat(normal, inner = (1, numqp))
    scaleareas = repeat([facedetjac], numqp)

    assemble_face_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        normals,
        conductivity,
        conductivity,
        jacobian,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_cut_cell_interelement_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    conductivity,
    mesh,
    cellsign,
    cellid,
    faceids,
    nbrfaceids,
    jacobian,
    facedetjac,
)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        quad1 = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if cellid < nbrcellid &&
           CutCellDG.solution_cell_id(mesh, cellsign, nbrcellid) !=
           CutCellDG.solution_cell_id(mesh, cellsign, cellid)

            nbrfaceid = nbrfaceids[faceid]
            nbrcellsign = CutCellDG.cell_sign(mesh, nbrcellid)

            if nbrcellsign == cellsign || nbrcellsign == 0
                quad2 = facequads[cellsign, nbrfaceid, nbrcellid]
                nodeids2 =
                    CutCellDG.nodal_connectivity(mesh, cellsign, nbrcellid)
                assemble_face_interelement_flux_operator!(
                    sysmatrix,
                    basis,
                    quad1,
                    quad2,
                    normals[faceid],
                    conductivity,
                    jacobian,
                    facedetjac[faceid],
                    nodeids1,
                    nodeids2,
                )
            end
        end
    end
end

function assemble_interelement_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    k1,
    k2,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    jacobian = CutCellDG.jacobian(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    nbrfaceids = [CutCellDG.opposite_face(faceid) for faceid in faceids]
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_interelement_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k1,
                mesh,
                +1,
                cellid,
                faceids,
                nbrfaceids,
                jacobian,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_interelement_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k2,
                mesh,
                -1,
                cellid,
                faceids,
                nbrfaceids,
                jacobian,
                facedetjac,
            )
        end
    end
end
################################################################################


################################################################################
function assemble_cell_interface_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    conductivity1,
    conductivity2,
    mesh,
    cellid,
    jacobian,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    normals = -1.0 * CutCellDG.interface_normals(interfacequads, cellid)
    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        normals,
        conductivity1,
        conductivity2,
        jacobian,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_interface_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    k1,
    k2,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    jacobian = CutCellDG.jacobian(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_interface_flux_operator!(
                sysmatrix,
                basis,
                interfacequads,
                k1,
                k2,
                mesh,
                cellid,
                jacobian,
            )
        end
    end
end
################################################################################



################################################################################
function assemble_boundary_face_flux_operator!(
    sysmatrix,
    basis,
    quad,
    normal,
    conductivity,
    jacobian,
    facedetjac,
    nodeids,
)

    numqp = length(quad)
    normals = repeat(normal, inner = (1, numqp))
    scaleareas = repeat([facedetjac], numqp)

    M = -1.0vec(
        transpose(
            flux_operator(
                basis,
                quad,
                quad,
                normals,
                conductivity,
                jacobian,
                scaleareas,
            ),
        ),
    )

    CutCellDG.assemble_couple_cell_matrix!(sysmatrix, nodeids, nodeids, 1, M)
end

function assemble_boundary_cell_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    conductivity,
    mesh,
    cellsign,
    cellid,
    faceids,
    jacobian,
    facedetjac,
)

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0 # this is a boundary face
            assemble_boundary_face_flux_operator!(
                sysmatrix,
                basis,
                quad,
                normals[faceid],
                conductivity,
                jacobian,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    k1,
    k2,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    jacobian = CutCellDG.jacobian(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_boundary_cell_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k1,
                mesh,
                +1,
                cellid,
                faceids,
                jacobian,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k2,
                mesh,
                -1,
                cellid,
                faceids,
                jacobian,
                facedetjac,
            )
        end
    end
end
################################################################################
