struct DGMesh1D
    cellmaps::Any
    nodalcoordinates::Any
    nodalconnectivity::Any
    cellconnectivity::Any
    numcells::Any
    numnodes::Any
    nodesperelement::Any
    xL::Any
    xR::Any
    interfacepoint::Any
    labels::Any
    elementsize
end

function DGMesh1D(xL, xR, interfacepoint, nelmts1, nelmts2, refpoints)
    dx1 = (interfacepoint - xL)/nelmts1
    dx2 = (xR - interfacepoint)/nelmts2
    elementsize = [dx1,dx2]

    c1 = construct_cell_maps(xL, interfacepoint, nelmts1)
    c2 = construct_cell_maps(interfacepoint, xR, nelmts2)
    cellmaps = vcat(c1, c2)

    nodalcoordinates = construct_nodal_coordinates(refpoints, cellmaps)

    numrefpoints = size(refpoints)[2]
    numcells = nelmts1 + nelmts2
    totalnumnodes = numrefpoints * numcells
    nodalconnectivity = construct_nodal_connectivity(numrefpoints, numcells)
    cellconnectivity = construct_cell_connectivity(numcells)

    labels = zeros(Int, numcells)
    labels[1:nelmts1] .= 1
    labels[(nelmts1+1):numcells] .= 2

    return DGMesh1D(
        cellmaps,
        nodalcoordinates,
        nodalconnectivity,
        cellconnectivity,
        numcells,
        totalnumnodes,
        numrefpoints,
        xL,
        xR,
        interfacepoint,
        labels,
        elementsize
    )
end

function Base.show(io::IO, dgmesh::DGMesh1D)
    xL = left_boundary(dgmesh)
    xR = right_boundary(dgmesh)
    nelmts = number_of_elements(dgmesh)
    nodesperelement = nodes_per_element(dgmesh)
    numnodes = number_of_nodes(dgmesh)

    str =
        "DGMesh1D\n\tLeft Boundary : $xL\n\t" *
        "Right Boundary : $xR\n\t" *
        "Num. Elements : $nelmts\n\t" *
        "Nodes/Element : $nodesperelement\n\t"
    "Num. Nodes : $numnodes"
    print(io, str)
end

function element_size(mesh::DGMesh1D)
    return mesh.elementsize
end

function cell_map(dgmesh::DGMesh1D,cellid)
    dgmesh.cellmaps[cellid]
end

function left_boundary(dgmesh::DGMesh1D)
    return dgmesh.xL
end

function right_boundary(dgmesh::DGMesh1D)
    return dgmesh.xR
end

function number_of_elements(dgmesh::DGMesh1D)
    return dgmesh.numcells
end

function number_of_cells(dgmesh::DGMesh1D)
    return dgmesh.numcells
end

function nodes_per_element(dgmesh::DGMesh1D)
    return dgmesh.nodesperelement
end

function number_of_nodes(dgmesh::DGMesh1D)
    return dgmesh.numnodes
end

function nodal_connectivity(dgmesh::DGMesh1D,cellid)
    return dgmesh.nodalconnectivity[:,cellid]
end

function nodal_coordinates(dgmesh::DGMesh1D)
    return dgmesh.nodalcoordinates
end

function element_label(dgmesh::DGMesh1D,cellid)
    return dgmesh.labels[cellid]
end

function construct_nodal_coordinates(refpoints, cellmaps)
    numrefpoints = size(refpoints)[2]
    numcells = length(cellmaps)
    totalnumnodes = numrefpoints * numcells

    nodalcoordinates = zeros(1, totalnumnodes)

    start = 1
    for cellid = 1:numcells
        nc = cellmaps[cellid](refpoints)
        stop = start + numrefpoints - 1

        nodalcoordinates[:, start:stop] .= nc
        start = stop + 1
    end
    return nodalcoordinates
end

function construct_nodal_connectivity(numrefpoints, numcells)
    totalnumnodes = numrefpoints * numcells
    nodalconnectivity = reshape(1:totalnumnodes, numrefpoints, :)
    return nodalconnectivity
end

function construct_cell_maps(xL, xR, ne)
    xrange = range(xL, stop = xR, length = ne + 1)
    cellmaps = []
    for cellid = 1:ne
        cellmap = CutCellDG.CellMap(xrange[cellid], xrange[cellid+1])
        push!(cellmaps, cellmap)
    end
    return cellmaps
end

function construct_cell_connectivity(numcells)
    cellconnectivity = zeros(Int, 2, numcells)
    for cellid = 1:numcells
        before = cellid == 1 ? 0 : cellid - 1
        after = cellid == numcells ? 0 : cellid + 1
        cellconnectivity[1, cellid] = before
        cellconnectivity[2, cellid] = after
    end
    return cellconnectivity
end
