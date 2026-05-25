using CairoMakie

function plot_graph(g; title = "graph", node_size = 7, edge_width = 0.6, edge_alpha = 0.20)
    S = permutedims(g.ndata.S)   # n × 2
    e = g.edata.e                # 7 × E

    fig = Figure(size = (700, 700))
    ax = Axis(fig[1, 1], aspect = DataAspect(), title = title)

    # edges
    for k in axes(e, 2)
        xi, yi = e[1, k], e[2, k]
        xj, yj = e[3, k], e[4, k]
        lines!(ax, [xi, xj], [yi, yj];
            color = (:gray, edge_alpha),
            linewidth = edge_width)
    end

    # nodes
    scatter!(ax, S[:, 1], S[:, 2];
        markersize = node_size)

    fig
end


function plot_node_neighborhood(g, node_id; node_size = 8)
    S = permutedims(g.ndata.S)   # n × 2
    e = g.edata.e                # 7 × E

    xi = S[node_id, 1]
    yi = S[node_id, 2]

    # 在当前实现里，e[1:2, k] 是 center node 的坐标
    idx = findall(k ->
        isapprox(e[1, k], xi; atol = 1e-8) &&
        isapprox(e[2, k], yi; atol = 1e-8),
        1:size(e, 2)
    )

    fig = Figure(size = (700, 700))
    ax = Axis(fig[1, 1], aspect = DataAspect(), title = "node $node_id neighborhood")

    # 全部点先淡淡画出来
    scatter!(ax, S[:, 1], S[:, 2]; markersize = 5, color = :gray80)

    # 该节点发出的边
    for k in idx
        xj, yj = e[3, k], e[4, k]
        lines!(ax, [xi, xj], [yi, yj];
            linewidth = 1.5)
        scatter!(ax, [xj], [yj]; markersize = node_size)
    end

    # 中心点
    scatter!(ax, [xi], [yi]; markersize = node_size + 4)

    fig
end





# using CairoMakie
using Statistics

function _coord_key(x, y; digits=8)
    return (round(Float64(x), digits=digits), round(Float64(y), digits=digits))
end

function _extract_graph_arrays(g; digits=8)
    S = permutedims(g.ndata.S)   # n × 2
    e = g.edata.e                # 7 × E

    n = size(S, 1)
    E = size(e, 2)

    # site -> node index
    coord_to_idx = Dict{Tuple{Float64,Float64}, Int}()
    for i in 1:n
        coord_to_idx[_coord_key(S[i,1], S[i,2]; digits=digits)] = i
    end

    src = Vector{Int}(undef, E)   # center node i
    dst = Vector{Int}(undef, E)   # neighbor node j
    dvec = Vector{Float64}(undef, E)

    for k in 1:E
        xi, yi = e[1,k], e[2,k]
        xj, yj = e[3,k], e[4,k]

        ki = _coord_key(xi, yi; digits=digits)
        kj = _coord_key(xj, yj; digits=digits)

        haskey(coord_to_idx, ki) || error("Cannot match center-node coordinate to site list.")
        haskey(coord_to_idx, kj) || error("Cannot match neighbor-node coordinate to site list.")

        src[k] = coord_to_idx[ki]
        dst[k] = coord_to_idx[kj]
        dvec[k] = Float64(e[7,k])
    end

    return S, src, dst, dvec
end

function graph_stats(g; digits=8)
    S, src, dst, dvec = _extract_graph_arrays(g; digits=digits)

    n = size(S, 1)
    E_directed = length(src)

    outdeg = zeros(Int, n)
    indeg  = zeros(Int, n)

    for k in eachindex(src)
        outdeg[src[k]] += 1
        indeg[dst[k]]  += 1
    end

    # undirected edge set
    undirected_edges = Set{Tuple{Int,Int}}()
    for k in eachindex(src)
        i, j = src[k], dst[k]
        i == j && continue
        a, b = min(i,j), max(i,j)
        push!(undirected_edges, (a,b))
    end

    E_undirected = length(undirected_edges)

    deg_undirected = zeros(Int, n)
    for (i,j) in undirected_edges
        deg_undirected[i] += 1
        deg_undirected[j] += 1
    end

    return (
        n = n,
        E_directed = E_directed,
        E_undirected = E_undirected,
        outdeg = outdeg,
        indeg = indeg,
        deg_undirected = deg_undirected,
        edge_lengths = dvec,
        mean_outdeg = mean(outdeg),
        mean_indeg = mean(indeg),
        mean_deg_undirected = mean(deg_undirected),
        min_edge_length = minimum(dvec),
        q25_edge_length = quantile(dvec, 0.25),
        median_edge_length = median(dvec),
        q75_edge_length = quantile(dvec, 0.75),
        max_edge_length = maximum(dvec),
    )
end

function plot_graph_info(
    g;
    title = "graph info",
    digits = 8,
    node_size = 7,
    edge_width = 0.6,
    edge_alpha = 0.18,
    hist_bins = 30,
    label_nodes = false,
)
    S, src, dst, dvec = _extract_graph_arrays(g; digits=digits)
    stats = graph_stats(g; digits=digits)

    # -----------------------------
    # Figure 1: full graph
    # -----------------------------
    fig_graph = Figure(size = (760, 760))
    ax = Axis(fig_graph[1,1], aspect = DataAspect(), title = title)

    for k in eachindex(src)
        i, j = src[k], dst[k]
        lines!(ax,
            [S[i,1], S[j,1]],
            [S[i,2], S[j,2]];
            color = (:gray, edge_alpha),
            linewidth = edge_width
        )
    end

    scatter!(ax, S[:,1], S[:,2]; markersize = node_size)

    if label_nodes
        for i in 1:size(S,1)
            text!(ax, S[i,1], S[i,2];
                text = string(i),
                fontsize = 10,
                align = (:left, :bottom))
        end
    end

    # -----------------------------
    # Figure 2: node degree heat
    # -----------------------------
    fig_deg = Figure(size = (760, 760))
    ax2 = Axis(fig_deg[1,1], aspect = DataAspect(), title = "undirected degree")

    for k in eachindex(src)
        i, j = src[k], dst[k]
        lines!(ax2,
            [S[i,1], S[j,1]],
            [S[i,2], S[j,2]];
            color = (:gray, 0.10),
            linewidth = 0.5
        )
    end

    sc = scatter!(ax2, S[:,1], S[:,2];
        color = stats.deg_undirected,
        markersize = node_size + 1)

    Colorbar(fig_deg[1,2], sc, label = "degree")

    # -----------------------------
    # Figure 3: edge length histogram
    # -----------------------------
    fig_len = Figure(size = (760, 520))
    ax3 = Axis(fig_len[1,1], title = "edge length distribution")
    hist!(ax3, dvec; bins = hist_bins)

    # -----------------------------
    # console summary
    # -----------------------------
    println("========================================")
    println("Graph summary")
    println("========================================")
    println("n nodes               = ", stats.n)
    println("E directed            = ", stats.E_directed)
    println("E undirected          = ", stats.E_undirected)
    println("mean out-degree       = ", round(stats.mean_outdeg, digits=3))
    println("mean in-degree        = ", round(stats.mean_indeg, digits=3))
    println("mean undirected degree= ", round(stats.mean_deg_undirected, digits=3))
    println("edge length min       = ", round(stats.min_edge_length, digits=4))
    println("edge length q25       = ", round(stats.q25_edge_length, digits=4))
    println("edge length median    = ", round(stats.median_edge_length, digits=4))
    println("edge length q75       = ", round(stats.q75_edge_length, digits=4))
    println("edge length max       = ", round(stats.max_edge_length, digits=4))
    println("========================================")

    return (
        stats = stats,
        fig_graph = fig_graph,
        fig_degree = fig_deg,
        fig_length = fig_len,
    )
end