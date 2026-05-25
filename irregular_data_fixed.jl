using NeuralEstimators
using Flux
using Flux: flatten, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
# using NNlib: scatter, gather
using NNlib
using Distances, Distributions
using Folds
using LinearAlgebra
using Optim
using Statistics
using AlgebraOfGraphics, CairoMakie
using CSV
using BSON: @save, @load
using SpecialFunctions           	# besselk, gamma
using StaticArrays
using SparseArrays
using Random

# using CUDA

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_types.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_generators_rbf.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/prior_experiment3.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/sim_experiment3.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Data_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Graph_nodewise.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_graph_functions.jl")


gridded = false
fixed_locs = true
N_rbf = 2
surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
    vary_rho1 = true,
    vary_rho2 = false,
    vary_delta = false,
    # rho2_const = 0.2,
    # delta_const = 0.0,
)

m_reps = 1
graph_k = 15
graph_r = 0.20f0
K_train = 2

train_parameters = prior(
    K_train,
    fixed_locs,
    surface_sampler;
    gridded = false,
    n_range = 200:300,
    λ_range = (10.0, 50.0),
    graph_k = graph_k,
    graph_r = graph_r,
);

Z_train = simulate(train_parameters, m_reps; gridded = false)

X_train, Y_train = build_supervised_dataset(
    train_parameters, Z_train;
    gridded = false,
    latent_targets = false,
    return_rho1 = true,
    return_rho2 = false,
    return_delta = false,
)



size(train_parameters.S)  # n×2
size(Z_train[1])             # n×m
g1 = X_train[1]
keys(g1.ndata)
keys(g1.edata)

size(Y_train[1])             # n×3
size(X_train[1].edata.e)
size(train_parameters.g.edata.e)

# =========================================================

# outputs
S = train_parameters.S      
surf = train_parameters.θ[1]   # SurfaceBundle
size(surf.sigma)
size(surf.rho1)  # n
X = Z_train[1]
size(X)  # n×2




fig1 = plot_replicate(
    S, X, surf;
    rep = 1,
    gridded = false,
    irregular_mode = :interp,
)

fig5 = plot_surface(
    S, surf;
    param = :all,
    gridded = false,
    irregular_mode = :interp,
    add_contour = true,
)

fig8 = plot_ellipses(
    S, surf;
    every = 2,
    scale = 0.60,
)


figg = plot_graph(train_parameters.g; title = "fixed irregular graph")
display(figg)

fig_local = plot_node_neighborhood(train_parameters.g, 2)
display(fig_local)


g1 = X_train[1]
S = permutedims(g1.ndata.S)         # n × 2
Z = dropdims(g1.ndata.Z; dims=(1,2))  # n, 因为你这里 m_reps = 1
fig = Figure(size = (700, 700))
ax = Axis(fig[1, 1], aspect = DataAspect(), title = "graph with node values")
# 先画边
e = g1.edata.e
for k in axes(e, 2)
    lines!(ax, [e[1,k], e[3,k]], [e[2,k], e[4,k]];
        color = (:gray, 0.12), linewidth = 0.5)
end
# 再按节点值上色
scatter!(ax, S[:,1], S[:,2];
    color = Z,
    markersize = 8)

display(fig)


# summary

# info = plot_graph_info(train_parameters.g; title = "fixed irregular graph")

# display(info.fig_graph)
# display(info.fig_degree)
# display(info.fig_length)

# info1 = plot_graph_info(X_train[1]; title = "sample graph")
# display(info1.fig_graph)
# display(info1.fig_degree)
# display(info1.fig_length)






include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Train&Pred_functions.jl")
# =========================================================
# 8) make train / val data
# =========================================================

gridded = false
fixed_locs = true
N_rbf = 2
m_reps = 10
graph_k = 15
graph_r = 0.20f0


K_train = 2000
K_val   = 200


surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
    vary_rho1 = true,
    vary_rho2 = true,
    vary_delta = true,
)

n_fixed = 300
λ_fixed = 30.0
S_fixed = maternclusterprocess(λ = λ_fixed, μ = n_fixed / λ_fixed)
g_fixed = nodegraph(S_fixed; k = graph_k, r = graph_r, random = false)

# train_parameters = prior(
#     K_train,
#     fixed_locs,
#     surface_sampler;
#     gridded = gridded,
#     n_range = 220:220,          # fixed irregular 时这个只决定那一套固定 sites 的规模
#     λ_range = (10.0, 50.0),
#     graph_k = graph_k,
# );
train_parameters = prior(
    K_train,
    fixed_locs,
    surface_sampler;
    gridded = false,
    graph_k = graph_k,
    graph_r = graph_r,
    S_fixed = S_fixed,
    g_fixed = g_fixed,
);
figg = plot_graph(train_parameters.g; title = "fixed irregular graph")
fig5 = plot_surface(
    S_fixed, train_parameters.θ[1];
    param = :all,
    gridded = false,
    irregular_mode = :interp,
    add_contour = true,
)


val_parameters = prior(
    K_val,
    fixed_locs,
    surface_sampler;
    gridded = false,
    graph_k = graph_k,
    graph_r = graph_r,
    S_fixed = S_fixed,
    g_fixed = g_fixed,
);
figg = plot_graph(val_parameters.g; title = "fixed irregular graph")



Z_train = simulate(train_parameters, m_reps; gridded = false)
Z_val   = simulate(val_parameters,   m_reps; gridded = false)

# X_train, Y_train = build_supervised_dataset(train_parameters, Z_train; gridded=false)
# X_val,   Y_val   = build_supervised_dataset(val_parameters,   Z_val;   gridded=false)
X_train, Y_train = build_supervised_dataset(
    train_parameters, Z_train;
    gridded = false,
    latent_targets = false,
    return_rho1 = true,
    return_rho2 = true,
    return_delta = true,
)

X_val, Y_val = build_supervised_dataset(
    val_parameters, Z_val;
    gridded = false,
    latent_targets = false,
    return_rho1 = true,
    return_rho2 = true,
    return_delta = true,
)


X_train[3]
keys(X_train[1].ndata)
size(X_train[1].ndata.Z)
Y_train[1]

# ---------------------------------- NBE-GNN
# include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/GNN_irr_fixed.jl")

# model = make_gnn_irr_fixed(
#     hidden = 64,
#     d_max = 0.25f0,
#     Kd = 8,
#     Kp = 16,
#     Kx = 8,
#     Ky = 8,
#     edge_width = 32,
#     edge_scale = 0.5f0,
# )

# model = train_gnn_irr_fixed!(
#     model,
#     X_train, Y_train,
#     X_val, Y_val;
#     epochs = 60,
#     lr = 3e-4,
#     λ_smooth = 1f-4,
#     seed = 1234,
#     verbose = true,
# )



# # # ---------------------------------- SGCN v2 shared body
# include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_irr_fixed_v2.jl")

# model = make_nodewise_sgcn(
#     hidden1 = 32,
#     hidden2 = 64,
#     hidden3 = 64,
#     K1 = 1,
#     K2 = 1,
#     K3 = 1,
#     head_hidden1 = 32,
#     head_hidden2 = 32,
#     activation = leakyrelu,
# )


# model = train_net!(
#     model,
#     X_train, Y_train,
#     X_val, Y_val;
#     loss_fn = node_loss_sgcn,
#     dataset_loss_fn = node_dataset_loss_sgcn,
#     epochs = 50,
#     lr = 1e-3,
#     λ_smooth = 1f-5,
#     seed = 1234,
#     verbose = true,
# )


# ---------------------------------- SGCN v3 multihead
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_irr_fixed_v3.jl")

model = make_nodewise_sgcn(
    hidden1 = 32,
    hidden2 = 64,
    hidden3 = 64,
    K1 = 8,
    K2 = 8,
    K3 = 8,
    head_hidden1 = 32,
    head_hidden2 = 32,
    activation = leakyrelu,
)


model = train_net!(
    model,
    X_train, Y_train,
    X_val, Y_val;
    loss_fn = node_loss_sgcn,
    dataset_loss_fn = node_dataset_loss_sgcn,
    epochs = 25,
    lr = 1e-3,
    λ_smooth = 1f-5,
    seed = 1234,
    verbose = true,
)

# =========================================================
# save trained model
# =========================================================

results_dir = "/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/irregular_data_fixed"
mkpath(results_dir)

model_path = joinpath(results_dir, "SGCN_v3(m=10).bson")
@save model_path model


# =========================================================
# test on one new irregular sample
# =========================================================

K_test = 1
test_parameters = prior(
    K_test,
    fixed_locs,
    surface_sampler;
    gridded = false,
    graph_k = graph_k,
    S_fixed = S_fixed,
    g_fixed = g_fixed,
);

Z_test = simulate(test_parameters, 100; gridded = false)
Z1 = Z_test[1]                       # n × m

g0 = get_graph(test_parameters, 1)   # fixed graph
S1 = get_sites(test_parameters, 1)
surf1 = get_surface(test_parameters, 1)

# rho1_hat, rho2_hat, delta_hat =
#     predict_nodewise_irr_fixed(model, g0, Z1)
# rho1_hat, rho2_hat, delta_hat =
#     predict_nodewise_sgcn(model, g0, Z1)
θhat = predict_nodewise_sgcn(model, g0, Z1)

# rho1_hat  = surf1.rho1
# rho2_hat = surf1.rho2
# delta_hat = surf1.delta
rho1_hat = vec(θhat[1, :])
rho2_hat = vec(θhat[2, :])
delta_hat = vec(θhat[3, :])


rho1_true  = surf1.rho1
rho2_true  = surf1.rho2
delta_true = surf1.delta



surf_hat = SurfaceBundle(
    surf1.sigma,
    rho1_hat,
    rho2_hat,
    delta_hat,
)

fig = plot_true_hat_surface_interp(
    S1,
    surf1,
    surf_hat;
    # which = (:delta,),
    colorrange_mode = :fixed,
)
display(fig)


# fig_true_hat = plot_true_hat_node_fields(
#     S1,
#     rho1_true, rho2_true, delta_true,
#     rho1_hat,  rho2_hat,  delta_hat;
#     marker = :circle,
#     markersize = 10,
# )
# display(fig_true_hat)

# fig_err = plot_three_node_fields(
#     S1,
#     rho1_hat .- rho1_true,
#     rho2_hat .- rho2_true,
#     delta_hat .- delta_true;
#     titles = ("rho1 error", "rho2 error", "delta error"),
# )
# display(fig_err)

# fig_true_interp = plot_surface(
#     S1,
#     surf1;
#     param = :all,
#     gridded = false,
#     irregular_mode = :interp,
#     add_contour = true,
#     interp_site_markersize = 4,
#     interp_site_color = :black,
#     interp_site_alpha = 1,
# )
# display(fig_true_interp)

# surf_hat = SurfaceBundle(
#     surf1.sigma,   # 或者 ones(Float32, length(rho1_hat))
#     rho1_hat,
#     rho2_hat,
#     delta_hat,
# )
# fig_hat_interp = plot_surface(
#     S1,
#     surf_hat;
#     param = :all,
#     gridded = false,
#     irregular_mode = :interp,
#     add_contour = true,
#     interp_site_markersize = 4,
#     interp_site_color = :black,
#     interp_site_alpha = 1,
# )
# display(fig_hat_interp)



# fig_true_ell = plot_ellipses(
#     S1,
#     surf1;
#     every = 2,
#     scale = 0.60,
# )
# display(fig_true_ell)




for test_seed in 1:10
    println("Running test_seed = ", test_seed)

    Random.seed!(test_seed)

    K_test = 1
    test_parameters = prior(
        K_test,
        fixed_locs,
        surface_sampler;
        gridded = false,
        graph_k = 12,
        S_fixed = S_fixed,
        g_fixed = g_fixed,
    )

    Z_test = simulate(test_parameters, 100; gridded = false)
    Z1 = Z_test[1]                       # n × m

    g0 = get_graph(test_parameters, 1)   # fixed graph
    S1 = get_sites(test_parameters, 1)
    surf1 = get_surface(test_parameters, 1)

    θhat = predict_nodewise_sgcn(model, g0, Z1)

    rho1_hat  = vec(θhat[1, :])
    rho2_hat  = vec(θhat[2, :])
    delta_hat = vec(θhat[3, :])

    surf_hat = SurfaceBundle(
        surf1.sigma,
        rho1_hat,
        rho2_hat,
        delta_hat,
    )

    fig = plot_true_hat_surface_interp(
        S1,
        surf1,
        surf_hat;
        # which = (:delta,),
        colorrange_mode = :fixed,
    )

    display(fig)

    fig_path = joinpath(results_dir, "test$(test_seed)(m=10).png")
    save(fig_path, fig)

    println("Saved figure to: ", fig_path)
end