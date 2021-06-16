function gradient_at_reference_points(
    nodalvalues,
    basis,
    refpoints,
    refcellids,
    levelsetsign,
    mesh,
)
    dim,numpts = size(refpoints)
    @assert length(refcellids) == numpts

    interpolatedgradients = zeros(dim,numpts)

    for idx in 1:numpts
        cellid = refcellids[idx]
        nodeids = CutCellDG.nodal_connectivity(mesh,levelsetsign,cellid)
        cellvals = nodalvalues[nodeids]

        grad = gradient(basis,refpoints[:,idx])

        interpolatedgradients[:,idx] = grad'*cellvals
    end
    return interpolatedgradients
end
