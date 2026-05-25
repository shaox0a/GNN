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
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/plot_graph_functions.jl")

include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model_functions.jl")


include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Train&Pred_functions.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/model-SGCN_irr_v3.jl")
# =========================================================
# 8) make train / val data
# =========================================================

gridded = false
fixed_locs = false
N_rbf = 2
m_reps = 100
process = :BR_MSP

graph_k = 15
graph_r = 0.20f0

K_train = 10
K_val   = 1


surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
    vary_rho1 = true,
    vary_rho2 = true,
    vary_delta = true,
    rho2_const = 0.2,
    delta_const = 0.0,
)



θ_train = prior(
    K_train,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
);
figg = plot_graph(θ_train.g[1]; title = "fixed irregular graph")
figθ = plot_surface(
    θ_train.S[2], θ_train.θ[2];
    param = :all,
    gridded = false,
    irregular_mode = :interp,
    add_contour = true,
)

θ_train.θ[1]

θ_val = prior(
    K_val,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
);
figg = plot_graph(θ_val.g[1]; title = "fixed irregular graph")

# -----------------------------------
# -----------------------------------
# -----------------------------------

network = make_nodewise_sgcn(
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

# estimator = PointEstimator(network)
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Architectures.jl")
include("/Users/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/Loss.jl")
network = NodewiseWrapper(network)
estimator = PointEstimator(network)


# CUDA.memory_status()
# GC.gc(true)
# CUDA.reclaim()
# CUDA.memory_status()
estimator = train(estimator, θ_train, θ_val, simulate_irregular; 
                  simulator_args = (m_reps, process,),
                  batchsize = 8,
                  loss = nodewise_mae_nbe)

# m_candidates = 1:100 #[1, 5, 10, 20, 50, 80, 100]
# estimator = train(estimator, θ_train, θ_val, simulate_irregular_wrapper; 
#                   simulator_args = (m_candidates,),
#                   batchsize = 8,
#                   loss = nodewise_mae_nbe)

                  
# -----------------------------------
# -----------------------------------
# -----------------------------------

# =========================================================
# Save trained estimator
# =========================================================

save_dir = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/irregular_data/"
mkpath(save_dir)

save_file = joinpath(save_dir, "nodewise_sgcn_varying_m.bson")

# Safer: save CPU version, avoiding CUDA serialization issues
estimator_save = cpu(estimator)

@save save_file estimator_save m_candidates graph_k graph_r gridded fixed_locs N_rbf

println("Saved estimator to: ", save_file)

# =========================================================
# Load trained estimator
# =========================================================

save_file = "/home/shaox0a/Desktop/GNN_nonstat/Codes/sitewise/results/irregular_data/nodewise_sgcn_varying_m.bson"

@load save_file estimator_save m_candidates graph_k graph_r gridded fixed_locs N_rbf

estimator = estimator_save

println("Loaded estimator from: ", save_file)

# =========================================================



K_test  = 100
θ_test = prior(
    K_test,
    fixed_locs,
    surface_sampler;
    gridded = gridded,
    graph_k = graph_k,
    graph_r = graph_r,
);
Z_test  = simulate_irregular(θ_test, 100, process)
θ_hat = NeuralEstimators.estimate(estimator, Z_test; batchsize = 1)


id = 1
Z1 = Z_test[id]                       # n × m
g1 = θ_test.g[id]
S1 = θ_test.S[id]
θ1 = θ_test.θ[id]


rho1_hat = θ_hat[id][1,:]
rho2_hat = θ_hat[id][2,:]
delta_hat = θ_hat[id][3,:]


rho1_true  = θ1.rho1
rho2_true  = θ1.rho2
delta_true = θ1.delta



θ1_hat = SurfaceBundle(
    θ1.sigma,
    rho1_hat,
    rho2_hat,
    delta_hat,
)

fig = plot_true_hat_surface_interp(
    S1,
    θ1,
    θ1_hat;
    # which = (:delta,),
    colorrange_mode = :fixed,
)
display(fig)


