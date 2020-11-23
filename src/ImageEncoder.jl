import Knet
using Knet: conv4, pool, mat, KnetArray, zeroone, param, param0, dropout, relu, batchnorm, bnmoments, bnparams, sigm
using HDF5
include("Layers.jl")

moments = [bnmoments(), bnmoments(), bnmoments()]
params = [param(bnparams(64)), param(bnparams(128)), param(bnparams(512))]

ImageEncoder = Chain(Conv(4,4,3,64),BNorm(64,moments[1],params[1],false),Conv(4,4,64,128),BNorm(128,moments[2],params[2],false),Conv(4,4,128,512),BNorm(512,moments[3],params[3],false))

function pool_features(x)
    pooled_features = sum(x, dims=[1,2])
end

obj_detector = Chain(Dense1(512,58))
