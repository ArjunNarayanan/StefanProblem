function gradient_at_reference_points(
    nodalvalues,
    basis,
    refpoints,
    refcellids,
    levelsetsign,
    mesh,
)
    dim, numpts = size(refpoints)
    @assert length(refcellids) == numpts
    jacobian = CutCellDG.jacobian(mesh)

    interpolatedgradients = zeros(dim, numpts)

    for idx = 1:numpts
        cellid = refcellids[idx]
        nodeids = CutCellDG.nodal_connectivity(mesh, levelsetsign, cellid)
        cellvals = nodalvalues[nodeids]

        grad = CutCellDG.transform_gradient(
            gradient(basis, refpoints[:, idx]),
            jacobian,
        )

        interpolatedgradients[:, idx] = grad' * cellvals
    end
    return interpolatedgradients
end

function interpolate_at_reference_points(
    nodalvalues,
    basis,
    refpoints,
    refcellids,
    levelsetsign,
    mesh,
)

    dim, numpts = size(refpoints)
    @assert length(refcellids) == numpts

    interpolatedvals = zeros(numpts)

    for idx = 1:numpts
        cellid = refcellids[idx]
        nodeids = CutCellDG.nodal_connectivity(mesh, levelsetsign, cellid)
        cellvals = nodalvalues[nodeids]

        interpolatedvals[idx] = cellvals' * basis(refpoints[:, idx])
    end
    return interpolatedvals
end
