function edge_switch(v0,normals)
    s = sign.(mapslices(sum,v0.*normals,dims=1))
    return 0.5*(s .* normals)
end
