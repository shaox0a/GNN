using NeuralEstimators
using Flux
using Flux: flatten, glorot_uniform
using GraphNeuralNetworks
using GraphNeuralNetworks: check_num_nodes
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
using CUDA

include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_types.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_generators_rbf.jl")

include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/prior_experiment3.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/sim_experiment3.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Data_functions.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Graph_nodewise.jl")

include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_functions.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_test_functions.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_graph_functions.jl")




include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Train&Pred_functions.jl")
# include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_irr_v3.0.jl")
# =========================================================
# 8) make train / val data
# =========================================================

gridded = true
fixed_locs = true
N_rbf = 2
m_reps = 100

graph_k = 15
graph_r = 0.20f0

K_train = 10000
K_val   = 1000
K_test  = 100



surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
    vary_rho1 = true,
    vary_rho2 = true,
    vary_delta = true,
    rho2_const = 0.2,
    delta_const = 0.0,
    model_type = :local_isotropy,
)


θ_train = prior(
    K_train,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
    model_type = :local_isotropy,
)
# figg = plot_graph(θ_train.g; title = "fixed gridded graph")
θ_train.θ[1].sigma
θ_train.θ[1].rho1
θ_train.θ[1].rho2
θ_train.θ[1].delta
θ_train.θ[1].nu



θ_val = prior(
    K_val,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
    model_type = :local_isotropy,
);
# figg = plot_graph(θ_val.g; title = "fixed gridded graph")



θ_test = prior(
    K_test,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
    model_type = :local_isotropy,
)
Z_test = simulate_irregular(θ_test, m_reps)




study1_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
mkpath(study1_dir)

study1_data_path = joinpath(study1_dir, "study1_test_gridded-data.bson")
@save study1_data_path θ_test Z_test
@load study1_data_path θ_test Z_test
# =========================================================
# MAP
# =========================================================
# include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/MAP_functions.jl")


# Π = make_nodewise_prior(θ_test.S)
# ξ = (Π = Π, S = θ_test.S)




# θ_map = MAP(Z_test, ξ)

# # using BSON: @save
# # save_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
# # isdir(save_dir) || mkpath(save_dir)
# # save_path = joinpath(save_dir, "theta_map_study1.bson")
# # @save save_path θ_map θ_test Z_test
# # load_path = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1/theta_map_study1.bson"
# # @load load_path θ_map θ_test Z_test


# id = 1
# Z1 = Z_test[id]
# g1 = θ_test.g isa Vector ? θ_test.g[id] : θ_test.g
# S1 = θ_test.S isa Vector ? θ_test.S[id] : θ_test.S
# θ1 = θ_test.θ[id]

# rho_map = vec(θ_map[id][1, :])
# delta_map = zeros(Float32, length(rho_map))

# θ1_map = SurfaceBundle(
#     θ1.sigma,
#     rho_map,
#     rho_map,
#     delta_map;
#     g = g1,
# )

# fig = plot_true_hat_surface_interp(
#     S1,
#     θ1,
#     θ1_map;
#     colorrange_mode = :fixed,
# )

# display(fig)


# --------------------------------------------------

# study1_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
# map_img_dir = joinpath(study1_dir, "MAP_images")
# mkpath(map_img_dir)

# for id in 1:10
#     Z1 = Z_test[id]
#     g1 = θ_test.g isa Vector ? θ_test.g[id] : θ_test.g
#     S1 = θ_test.S isa Vector ? θ_test.S[id] : θ_test.S
#     θ1 = θ_test.θ[id]

#     θ_map_1 = MAP(Z1, S1, Π)

#     rho_map = vec(θ_map_1[1, :])
#     delta_map = zeros(Float32, length(rho_map))

#     θ1_map = SurfaceBundle(
#         θ1.sigma,
#         rho_map,
#         rho_map,
#         delta_map;
#         g = g1,
#     )

#     fig = plot_true_hat_surface_interp(
#         S1,
#         θ1,
#         θ1_map;
#         which = (:rho1,),
#         colorrange_mode = :fixed,
#     )

#     mae_rho = mean(abs.(rho_map .- Float32.(θ1.rho1)))
#     println("MAP id = $id, rho MAE = $mae_rho")

#     save(joinpath(map_img_dir, "MAP_id$(id).png"), fig)
# end



# =========================================================
# CNN
# =========================================================
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-CNN_study1.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Loss_study1.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/IntervalEstimator1.jl")


network = make_cnn_study1(
    hidden = 64,
    pool = :mean,
)
interval_network = make_cnn_study1_interval(
    hidden = 64,
    pool = :mean,
)

estimator = PointEstimator(network)
unc_estimator = IntervalEstimator1(interval_network)


CUDA.memory_status()
GC.gc(true)
CUDA.reclaim()
CUDA.memory_status()
estimator = train(
    estimator,
    θ_train,
    θ_val,
    simulate_grided;
    simulator_args = (m_reps,),
    simulator_kwargs = (; grid_shape = (16, 16)),
    batchsize = 8,
    loss = nodewise_mae_nbe_study1,
    epochs = 100,
)
unc_estimator = train(
    unc_estimator,
    θ_train,
    θ_val,
    simulate_grided;
    simulator_args = (m_reps,),
    simulator_kwargs = (; grid_shape = (16, 16)),
    batchsize = 8,
    loss = nodewise_interval_loss_study1,
    epochs = 100,
)


# --------
cnn_estimator_path = joinpath(study1_dir, "cnn_study1_gridded_estimator.bson")
@save cnn_estimator_path estimator
cnn_unc_estimator_path = joinpath(study1_dir, "cnn_study1_gridded_unc_estimator.bson")
@save cnn_unc_estimator_path unc_estimator
# @load cnn_unc_estimator_path unc_estimator
# --------


H, W = 16, 16
Z_test_cnn = map(Z_test) do g
    Z = Array(g.ndata.Z)                         # 1 × m × n
    Z = permutedims(dropdims(Z, dims = 1), (2, 1))  # n × m
    Float32.(reshape(Z, H, W, 1, size(Z, 2)))       # H × W × 1 × m
end

θ_hat = NeuralEstimators.estimate(
    estimator,
    Z_test_cnn;
    batchsize = K_test,
)
θ_hat[1]          # 1 × 256
θ_hat[1][1, :]    # estimated rho surface vector

# First estimate uncertainty intervals
θ_hat_unc = NeuralEstimators.estimate(
    unc_estimator,
    Z_test_cnn;
    batchsize = K_test,
)

# θ_hat_unc[id] is 2 × n:
# row 1 = lower
# row 2 = upper

cnn_img_dir = joinpath(study1_dir, "CNN_gridded_images")
mkpath(cnn_img_dir)

for id in 1:10
    S1 = get_sites(θ_test, id)
    g1 = get_graph(θ_test, id)
    θ1 = get_surface(θ_test, id)

    rho_true = Float32.(θ1.rho1)

    # ==================================================
    # Point estimate
    # ==================================================
    rho_hat = Float32.(vec(θ_hat[id][1, :]))

    mae_rho = mean(abs.(rho_hat .- rho_true))
    println("id = $id, rho MAE = $mae_rho")

    # ==================================================
    # Interval estimate
    # ==================================================
    rho_lower = Float32.(vec(θ_hat_unc[id][1, :]))
    rho_upper = Float32.(vec(θ_hat_unc[id][2, :]))

    rho_width = rho_upper .- rho_lower
    cover_vec = (rho_true .>= rho_lower) .& (rho_true .<= rho_upper)

    coverage_id = mean(cover_vec)
    mean_width_id = mean(rho_width)

    println("id = $id, interval coverage = $coverage_id")
    println("id = $id, mean interval width = $mean_width_id")

    # ==================================================
    # Build grids
    # ==================================================
    xg, yg, Z_true  = gridify(S1, rho_true)
    _,  _,  Z_hat   = gridify(S1, rho_hat)
    _,  _,  Z_upper = gridify(S1, rho_upper)
    _,  _,  Z_lower = gridify(S1, rho_lower)
    _,  _,  Z_width = gridify(S1, rho_width)

    # same color range for true / estimate / upper / lower
    cr_main = (
        minimum(vcat(vec(Z_true), vec(Z_hat), vec(Z_upper), vec(Z_lower))),
        maximum(vcat(vec(Z_true), vec(Z_hat), vec(Z_upper), vec(Z_lower)))
    )

    cr_width = (
        minimum(vec(Z_width)),
        maximum(vec(Z_width))
    )

    # ==================================================
    # Combined figure: true | estimate | upper | lower | width
    # ==================================================
    fig_all = Figure(size = (2200, 500))

    ax1 = Axis(
        fig_all[1, 1],
        aspect = DataAspect(),
        title = "True surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm1 = heatmap!(ax1, xg, yg, Z_true; colorrange = cr_main, colormap = :viridis)

    ax2 = Axis(
        fig_all[1, 2],
        aspect = DataAspect(),
        title = "Estimated surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm2 = heatmap!(ax2, xg, yg, Z_hat; colorrange = cr_main, colormap = :viridis)

    ax3 = Axis(
        fig_all[1, 3],
        aspect = DataAspect(),
        title = "Upper surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm3 = heatmap!(ax3, xg, yg, Z_upper; colorrange = cr_main, colormap = :viridis)

    ax4 = Axis(
        fig_all[1, 4],
        aspect = DataAspect(),
        title = "Lower surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm4 = heatmap!(ax4, xg, yg, Z_lower; colorrange = cr_main, colormap = :viridis)

    ax5 = Axis(
        fig_all[1, 5],
        aspect = DataAspect(),
        title = "Width surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm5 = heatmap!(ax5, xg, yg, Z_width; colorrange = cr_width, colormap = :viridis)

    Colorbar(fig_all[1, 6], hm1, label = "rho")
    Colorbar(fig_all[1, 7], hm5, label = "width")

    save(joinpath(cnn_img_dir, "CNN_all_surfaces_id$(id).png"), fig_all)

    # ==================================================
    # Coverage map
    # black = covered
    # red   = not covered
    # ==================================================
    fig_cover = Figure(size = (700, 600))
    ax_cover = Axis(
        fig_cover[1, 1],
        aspect = DataAspect(),
        title = "CI coverage, id = $id",
        xlabel = "x",
        ylabel = "y",
    )

    cover_colors = ifelse.(cover_vec, :black, :red)

    scatter!(
        ax_cover,
        S1[:, 1],
        S1[:, 2];
        color = cover_colors,
        markersize = 10,
    )

    save(joinpath(cnn_img_dir, "CNN_interval_cover_id$(id).png"), fig_cover)
end
# mae_rho = mean(abs.(rho_hat .- rho_true))
# println("Test id = $id")
# println("rho MAE = $mae_rho")





# =========================================================
# GNN
# =========================================================
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_study1.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Architectures.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Loss.jl")


network = make_nodewise_sgcn_study1(
    hidden1 = 32,
    hidden2 = 64,
    hidden3 = 64,
    K1 = 1,
    K2 = 1,
    K3 = 1,
    head_hidden1 = 32,
    head_hidden2 = 32,
    activation = leakyrelu,
)
interval_network = make_nodewise_sgcn_study1_interval(
    hidden1 = 32,
    hidden2 = 64,
    hidden3 = 64,
    K1 = 1,
    K2 = 1,
    K3 = 1,
    head_hidden1 = 32,
    head_hidden2 = 32,
    activation = leakyrelu,
)


network = NodewiseWrapper(network)
interval_network = NodewiseWrapper(interval_network)

estimator = PointEstimator(network)
unc_estimator = IntervalEstimator1(interval_network)


CUDA.memory_status()
GC.gc(true)
CUDA.reclaim()
CUDA.memory_status()
estimator = train(estimator, θ_train, θ_val, simulate_irregular; 
                  simulator_args = (m_reps,),
                  batchsize = 8,
                  loss = nodewise_mae_nbe_study1)
unc_estimator = train( unc_estimator, θ_train, θ_val, simulate_irregular;
    simulator_args = (m_reps,),
    batchsize = 8,
    loss = nodewise_interval_loss_study1,
    epochs = 100,
)


# --------
gnn_estimator_path = joinpath(study1_dir, "gnn_study1_gridded_estimator.bson")
@save gnn_estimator_path estimator
@load gnn_estimator_path estimator

gnn_unc_estimator_path = joinpath(study1_dir, "gnn_study1_gridded_unc_estimator.bson")
@save gnn_unc_estimator_path unc_estimator
@load gnn_unc_estimator_path unc_estimator
# --------


θ_hat = NeuralEstimators.estimate(
    estimator,
    Z_test;
    batchsize = K_test,
)
θ_hat[1]          # 1 × 256
θ_hat[1][1, :]    # estimated rho surface vector

θ_hat_unc = NeuralEstimators.estimate(
    unc_estimator,
    Z_test;
    batchsize = K_test,
)
θ_hat_unc[1]          # 2 × 256
θ_hat_unc[1][1, :]    # lower rho surface
θ_hat_unc[1][2, :]    # upper rho surface

gnn_img_dir = joinpath(study1_dir, "GNN_gridded_images")
mkpath(gnn_img_dir)
for id in 1:10
    S1 = get_sites(θ_test, id)
    g1 = get_graph(θ_test, id)
    θ1 = get_surface(θ_test, id)

    # ==================================================
    # Point estimate
    # ==================================================
    rho_true = Float32.(θ1.rho1)
    rho_hat  = Float32.(vec(θ_hat[id][1, :]))

    mae_rho = mean(abs.(rho_hat .- rho_true))
    println("id = $id, rho MAE = $mae_rho")

    # ==================================================
    # Interval estimate
    # ==================================================
    rho_lower = Float32.(vec(θ_hat_unc[id][1, :]))
    rho_upper = Float32.(vec(θ_hat_unc[id][2, :]))

    rho_width = rho_upper .- rho_lower
    cover_vec = (rho_true .>= rho_lower) .& (rho_true .<= rho_upper)

    coverage_id = mean(cover_vec)
    mean_width_id = mean(rho_width)

    println("id = $id, interval coverage = $coverage_id")
    println("id = $id, mean interval width = $mean_width_id")

    # ==================================================
    # Build grids
    # ==================================================
    xg, yg, Z_true  = gridify(S1, rho_true)
    _,  _,  Z_hat   = gridify(S1, rho_hat)
    _,  _,  Z_upper = gridify(S1, rho_upper)
    _,  _,  Z_lower = gridify(S1, rho_lower)
    _,  _,  Z_width = gridify(S1, rho_width)

    cr_main = (
        minimum(vcat(vec(Z_true), vec(Z_hat), vec(Z_upper), vec(Z_lower))),
        maximum(vcat(vec(Z_true), vec(Z_hat), vec(Z_upper), vec(Z_lower)))
    )

    cr_width = (
        minimum(vec(Z_width)),
        maximum(vec(Z_width))
    )

    # ==================================================
    # Combined figure:
    # true | estimated | upper | lower | width
    # ==================================================
    fig_all = Figure(size = (2200, 500))

    ax1 = Axis(
        fig_all[1, 1],
        aspect = DataAspect(),
        title = "True surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm1 = heatmap!(ax1, xg, yg, Z_true; colorrange = cr_main, colormap = :viridis)

    ax2 = Axis(
        fig_all[1, 2],
        aspect = DataAspect(),
        title = "Estimated surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm2 = heatmap!(ax2, xg, yg, Z_hat; colorrange = cr_main, colormap = :viridis)

    ax3 = Axis(
        fig_all[1, 3],
        aspect = DataAspect(),
        title = "Upper surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm3 = heatmap!(ax3, xg, yg, Z_upper; colorrange = cr_main, colormap = :viridis)

    ax4 = Axis(
        fig_all[1, 4],
        aspect = DataAspect(),
        title = "Lower surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm4 = heatmap!(ax4, xg, yg, Z_lower; colorrange = cr_main, colormap = :viridis)

    ax5 = Axis(
        fig_all[1, 5],
        aspect = DataAspect(),
        title = "Width surface",
        xlabel = "x",
        ylabel = "y",
    )
    hm5 = heatmap!(ax5, xg, yg, Z_width; colorrange = cr_width, colormap = :viridis)

    Colorbar(fig_all[1, 6], hm1, label = "rho")
    Colorbar(fig_all[1, 7], hm5, label = "width")

    save(joinpath(gnn_img_dir, "GNN_all_surfaces_id$(id).png"), fig_all)

    # ==================================================
    # Coverage map
    # black = covered
    # red   = not covered
    # ==================================================
    fig_cover = Figure(size = (700, 600))
    ax_cover = Axis(
        fig_cover[1, 1],
        aspect = DataAspect(),
        title = "CI coverage, id = $id",
        xlabel = "x",
        ylabel = "y",
    )

    cover_colors = ifelse.(cover_vec, :black, :red)

    scatter!(
        ax_cover,
        S1[:, 1],
        S1[:, 2];
        color = cover_colors,
        markersize = 10,
    )

    save(joinpath(gnn_img_dir, "GNN_interval_cover_id$(id).png"), fig_cover)
end