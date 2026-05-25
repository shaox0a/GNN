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
# using CUDA

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_types.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Surface_generators_rbf.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/prior_experiment3.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/sim_experiment3.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Data_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Graph_nodewise.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_test_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_graph_functions.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Train&Pred_functions.jl")
# include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_irr_v3.0.jl")

# =========================================================
# 8) make train / val data
# =========================================================

gridded = true
fixed_locs = true
N_rbf = 2
m_reps = 1

graph_k = 15
graph_r = 0.20f0

K_train = 2
K_val   = 1
K_test  = 1

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
)

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

study1_dir = "/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
mkpath(study1_dir)

study1_data_path = joinpath(study1_dir, "study1_test_gridded-data.bson")
@save study1_data_path θ_test Z_test
@load study1_data_path θ_test Z_test

# =========================================================
# MAP
# =========================================================

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/MAP_functions.jl")

Π = make_nodewise_prior(θ_test.S)
ξ = (Π = Π, S = θ_test.S, g = θ_test.g)

# MAP options
# λ_ridge_map controls eta-ridge penalty:
#   loss = -loglik + λ_ridge_map * ||eta - eta0||^2
# Set λ_ridge_map = 0.0 to turn ridge off.
#
# λ_smoo_map controls eta-smoothness penalty:
#   smoo_weight_map = :inv_distance uses w_ij = 1 / distance(s_i, s_j)
#   smoo_weight_map = :unit uses w_ij = 1
# Set λ_smoo_map = 0.0 to turn smoothness off.
λ_ridge_map = 1.0
λ_smoo_map = 1.0
smoo_weight_map = :inv_distance
use_stationary_init_map = true

result_map = @timed MAP(
    Z_test,
    ξ;
    λ_ridge = λ_ridge_map,
    λ_smoo = λ_smoo_map,
    smoo_weight = smoo_weight_map,
    use_stationary_init = use_stationary_init_map,
)

θ_map = result_map.value
println("MAP time = ", result_map.time, " seconds")
println("MAP memory = ", result_map.bytes / 1024^2, " MB")

# using BSON: @save
# save_dir = "/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
# isdir(save_dir) || mkpath(save_dir)
# save_path = joinpath(save_dir, "theta_map_study1.bson")
# @save save_path θ_map θ_test Z_test
# load_path = "/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1/theta_map_study1.bson"
# @load load_path θ_map θ_test Z_test

id = 1
Z1 = Z_test[id]
g1 = θ_test.g isa Vector ? θ_test.g[id] : θ_test.g
S1 = θ_test.S isa Vector ? θ_test.S[id] : θ_test.S
θ1 = θ_test.θ[id]

rho_map = vec(θ_map[id][1, :])
delta_map = zeros(Float32, length(rho_map))

θ1_map = SurfaceBundle(
    θ1.sigma,
    rho_map,
    rho_map,
    delta_map;
    g = g1,
)

fig = plot_true_hat_surface_interp(
    S1,
    θ1,
    θ1_map;
    colorrange_mode = :fixed,
)

display(fig)

# --------------------------------------------------

study1_dir = "/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/study1"
map_img_dir = joinpath(study1_dir, "MAP_images")
mkpath(map_img_dir)

for id in eachindex(Z_test)
    Z1 = Z_test[id]
    g1 = θ_test.g isa Vector ? θ_test.g[id] : θ_test.g
    S1 = θ_test.S isa Vector ? θ_test.S[id] : θ_test.S
    θ1 = θ_test.θ[id]

    Π1 = _prior_for_k(Π, id)

    θ_map_1 = MAP(
        Z1,
        S1,
        Π1,
        g1;
        λ_ridge = λ_ridge_map,
        λ_smoo = λ_smoo_map,
        smoo_weight = smoo_weight_map,
        use_stationary_init = use_stationary_init_map,
    )

    rho_map = vec(θ_map_1[1, :])
    delta_map = zeros(Float32, length(rho_map))

    θ1_map = SurfaceBundle(
        θ1.sigma,
        rho_map,
        rho_map,
        delta_map;
        g = g1,
    )

    fig = plot_true_hat_surface_interp(
        S1,
        θ1,
        θ1_map;
        which = (:rho1,),
        colorrange_mode = :fixed,
    )

    mae_rho = mean(abs.(rho_map .- Float32.(θ1.rho1)))
    println("MAP id = $id, rho MAE = $mae_rho")

    save(joinpath(map_img_dir, "MAP_id$(id).png"), fig)
end