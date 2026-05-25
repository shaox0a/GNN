using NeuralEstimators
using Flux
using Flux: flatten
using GraphNeuralNetworks
using Distances, Distributions
using Folds
using LinearAlgebra
using Optim
using Statistics
using AlgebraOfGraphics, CairoMakie
using CSV
using BSON: @save, @load
using SpecialFunctions            # besselk, gamma
using StaticArrays
using SparseArrays
# using CUDA

# =========================================================
# load local files
# =========================================================

const BASE_DIR = @__DIR__

include(joinpath(BASE_DIR, "Surface_types.jl"))
include(joinpath(BASE_DIR, "Surface_generators_rbf.jl"))
include(joinpath(BASE_DIR, "prior_experiment3.jl"))
include(joinpath(BASE_DIR, "Graph_nodewise.jl"))
include(joinpath(BASE_DIR, "sim_experiment3.jl"))
include(joinpath(BASE_DIR, "Data_functions.jl"))
include(joinpath(BASE_DIR, "plot_functions.jl"))


# =========================================================
# 1) quick sanity check: one gridded sample + plotting
# =========================================================

fixed_locs = true
N_rbf = 2

surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
)

# generate 1 nonstationary GP setting on the fixed 16×16 grid
train_parameters = prior(
    1,
    fixed_locs,
    surface_sampler;
    gridded = true,
    grid_side = 16,
);

# generate 100 replicates
Z = simulate(train_parameters, 100; gridded = true, grid_shape = (16, 16))

# outputs under the new surface-only interface
S = get_sites(train_parameters, 1)          # 256×2
surf = get_surface(train_parameters, 1)     # SurfaceBundle
L = train_parameters.chols[1]               # Cholesky factor
X = Z[1]                                    # 16×16×1×100

fig1 = plot_replicate(S, X, surf; rep = 1, gridded = true)

a = plot_replicate(
    S, X, surf;
    rep = 3,
    gridded = true,
    add_contour = true,
    add_ellipses = true,
    ellipse_every = 2,
    ellipse_scale = 0.60,
)
display(a)

b = plot_surface(S, surf; param = :all, gridded = true, add_contour = true)
display(b)

c = plot_ellipses(S, surf; every = 2, scale = 0.60)
display(c)

# =========================================================
# 2) make train / val data for gridded CNN
# =========================================================

include(joinpath(BASE_DIR, "Train&Pred_functions.jl"))

fixed_locs = true
N_rbf = 2

K_train = 1000
K_val   = 100
m_reps  = 1

surface_sampler = make_rbf_surface_sampler(
    N_rbf = N_rbf,
    Πb = Π_base,
)

train_parameters = prior(
    K_train,
    fixed_locs,
    surface_sampler;
    gridded = true,
    grid_side = 16,
);

val_parameters = prior(
    K_val,
    fixed_locs,
    surface_sampler;
    gridded = true,
    grid_side = 16,
);

Z_train = simulate(train_parameters, m_reps; gridded = true, grid_shape = (16, 16))
Z_val   = simulate(val_parameters,   m_reps; gridded = true, grid_shape = (16, 16))

X_train, Y_train = build_supervised_dataset(train_parameters, Z_train; gridded = true)
X_val,   Y_val   = build_supervised_dataset(val_parameters,   Z_val;   gridded = true)

size(Y_train)
size(Y_train[1])
size(Z_train[1])
size(X_train[1])
X_train[1][:, :, :, 1]
X_train[1]

# ------------------------------------- CNN
include(joinpath(BASE_DIR, "model-CNN_multihead_grid.jl"))
# model = make_surface_net(hidden = 64)
model = make_surface_net_multihead(; in_ch=1, hidden = 64)

model = train_net!(
    model,
    X_train, Y_train,
    X_val, Y_val;
    loss_fn = surface_loss,
    dataset_loss_fn = dataset_loss,
    epochs = 40,
    lr = 1e-3,
    λ_smooth = 1f-4,
    seed = 1234,
    verbose = true,
)


# # ------------------------------------- Transformer
# include(joinpath(BASE_DIR, "TF_naive.jl"))

# S_train = get_sites(train_parameters, 1)
# S_val   = get_sites(val_parameters, 1)

# model = make_surface_tf_coordconcat(
#     grid_shape = (16, 16),
#     in_ch = size(X_train[1], 3),
#     d_model = 64,
#     nheads = 4,
#     depth = 4,
#     mlp_dim = 128,
#     dropout = 0f0,
# )

# model = train_net_tf!(
#     model,
#     X_train, Y_train, S_train,
#     X_val, Y_val, S_val;
#     loss_fn = surface_loss_tf,
#     dataset_loss_fn = dataset_loss_tf,
#     epochs = 40,
#     lr = 1e-3,
#     λ_smooth = 0f0,
#     seed = 1234,
#     verbose = true,
# )


# =========================================================
# 3) test prediction on a fresh gridded sample
# =========================================================

N_rbf1 = 2
surface_sampler1 = make_rbf_surface_sampler(
    N_rbf = N_rbf1,
    Πb = Π_base,
)

parameters1 = prior(
    1,
    true,
    surface_sampler1;
    gridded = true,
    grid_side = 16,
)

Z1 = simulate(parameters1, 100; gridded = true, grid_shape = (16, 16))

S1 = get_sites(parameters1, 1)
surf1 = get_surface(parameters1, 1)
X1 = Z1[1]

rho1_surface_true, rho2_surface_true, delta_surface_true =
    extract_true_surfaces(S1, surf1)

rho1_surface_hat, rho2_surface_hat, delta_surface_hat =
    predict_surfaces(model, X1, S1)
# rho1_surface_hat, rho2_surface_hat, delta_surface_hat =
#     predict_surfaces_tf(model, X1, S1)

fig_true = plot_three_surfaces(
    rho1_surface_true, rho2_surface_true, delta_surface_true;
    titles = ("rho1 true", "rho2 true", "delta true"),
)
display(fig_true)

fig_hat = plot_three_surfaces(
    rho1_surface_hat, rho2_surface_hat, delta_surface_hat;
    titles = ("rho1 hat", "rho2 hat", "delta hat"),
)
display(fig_hat)

fig_err = plot_three_surfaces(
    rho1_surface_hat .- rho1_surface_true,
    rho2_surface_hat .- rho2_surface_true,
    delta_surface_hat .- delta_surface_true;
    titles = ("rho1 error", "rho2 error", "delta error"),
)
display(fig_err)
