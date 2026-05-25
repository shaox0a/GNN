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

gridded = false
fixed_locs = false
N_rbf = 2
m_reps = 1
graph_k = 15
graph_r = 0.20f0

K_train = 10000
K_val   = 1000
K_test  = 100

study1_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
mkpath(study1_dir)


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

Z_test = simulate_irregular(θ_test, 100)

study1_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
mkpath(study1_dir)

study1_data_path = joinpath(study1_dir, "study1_test_irregular-data.bson")
@save study1_data_path θ_test Z_test
@load study1_data_path θ_test Z_test

# =========================================================
# MAP
# =========================================================
# include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/MAP_functions.jl")


# Π = make_nodewise_prior(θ_test.S)
# ξ = (Π = Π, S = θ_test.S)
# θ_map_1 = MAP(Z_test[1], θ_test.S, Π)
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



# =========================================================
# GNN
# =========================================================
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_study1.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Architectures.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Loss_study1.jl")
include("/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/IntervalEstimator1.jl")


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


estimator = train(
    estimator,
    θ_train,
    θ_val,
    simulate_irregular;
    simulator_args = (m_reps,),
    batchsize = 8,
    loss = nodewise_mae_nbe_study1,
    epochs = 100,
    use_gpu = true,
)

unc_estimator = train(
    unc_estimator,
    θ_train,
    θ_val,
    simulate_irregular;
    simulator_args = (m_reps,),
    batchsize = 8,
    loss = nodewise_interval_loss_study1,
    epochs = 100,
    use_gpu = true,
)


# --------
gnn_estimator_path = joinpath(study1_dir, "gnn_study1_irregular_estimator.bson")
@save gnn_estimator_path estimator
# @load gnn_estimator_path estimator

gnn_unc_estimator_path = joinpath(study1_dir, "gnn_study1_irregular_unc_estimator.bson")
@save gnn_unc_estimator_path unc_estimator
# @load gnn_unc_estimator_path unc_estimator
# --------


θ_hat = NeuralEstimators.estimate(
    estimator,
    Z_test;
    batchsize = K_test,
)

θ_hat[1]          # 1 × n
θ_hat[1][1, :]    # estimated rho surface vector


θ_hat_unc = NeuralEstimators.estimate(
    unc_estimator,
    Z_test;
    batchsize = K_test,
)

θ_hat_unc[1]          # 2 × n
θ_hat_unc[1][1, :]    # lower rho surface
θ_hat_unc[1][2, :]    # upper rho surface

gnn_img_dir = joinpath(study1_dir, "GNN_irregular_images")
mkpath(gnn_img_dir)

for id in 1:min(10, K_test)
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
    # Color ranges
    # ==================================================
    cr_main = (
        minimum(vcat(rho_true, rho_hat, rho_upper, rho_lower)),
        maximum(vcat(rho_true, rho_hat, rho_upper, rho_lower))
    )

    if cr_main[1] == cr_main[2]
        cr_main = (cr_main[1] - 1f-6, cr_main[2] + 1f-6)
    end

    cr_width = (
        minimum(rho_width),
        maximum(rho_width)
    )

    if cr_width[1] == cr_width[2]
        cr_width = (cr_width[1] - 1f-6, cr_width[2] + 1f-6)
    end

    # ==================================================
    # Combined figure for irregular data:
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
    sc1 = scatter!(
        ax1,
        S1[:, 1],
        S1[:, 2];
        color = rho_true,
        colorrange = cr_main,
        colormap = :viridis,
        markersize = 10,
    )

    ax2 = Axis(
        fig_all[1, 2],
        aspect = DataAspect(),
        title = "Estimated surface",
        xlabel = "x",
        ylabel = "y",
    )
    sc2 = scatter!(
        ax2,
        S1[:, 1],
        S1[:, 2];
        color = rho_hat,
        colorrange = cr_main,
        colormap = :viridis,
        markersize = 10,
    )

    ax3 = Axis(
        fig_all[1, 3],
        aspect = DataAspect(),
        title = "Upper surface",
        xlabel = "x",
        ylabel = "y",
    )
    sc3 = scatter!(
        ax3,
        S1[:, 1],
        S1[:, 2];
        color = rho_upper,
        colorrange = cr_main,
        colormap = :viridis,
        markersize = 10,
    )

    ax4 = Axis(
        fig_all[1, 4],
        aspect = DataAspect(),
        title = "Lower surface",
        xlabel = "x",
        ylabel = "y",
    )
    sc4 = scatter!(
        ax4,
        S1[:, 1],
        S1[:, 2];
        color = rho_lower,
        colorrange = cr_main,
        colormap = :viridis,
        markersize = 10,
    )

    ax5 = Axis(
        fig_all[1, 5],
        aspect = DataAspect(),
        title = "Width surface",
        xlabel = "x",
        ylabel = "y",
    )
    sc5 = scatter!(
        ax5,
        S1[:, 1],
        S1[:, 2];
        color = rho_width,
        colorrange = cr_width,
        colormap = :viridis,
        markersize = 10,
    )

    Colorbar(fig_all[1, 6], sc1, label = "rho")
    Colorbar(fig_all[1, 7], sc5, label = "width")

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