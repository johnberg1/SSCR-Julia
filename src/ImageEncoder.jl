import Knet
using Knet: conv4, pool, mat, KnetArray, zeroone, param, param0, dropout, relu, batchnorm, bnmoments, bnparams, sigm
using HDF5

# Need to fix batchnorm layer
function simpleRead(filePath,exIdx)
    fid = h5open(filePath,"r")
    example = fid["$(exIdx)"]
    images = read(example["images"])
    utterences = read(example["utterences"])
    objects = read(example["objects"])
    coords = read(example["coords"])
    scene_id = read(example["scene_id"])
    (coords,images,objects,scene_id,utterences)
end

struct Conv; w; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, padding = 1, stride = 2))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Conv(param(w1,w2,cx,cy),f)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=sigm;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

struct BNorm; C; end
(bn::BNorm)(x) = batchnorm(x ,bnmoments(),param(bnparams(bn.C)))

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

net = Chain(Conv(4,4,3,64),BNorm(64),Conv(4,4,64,128),BNorm(128),Conv(4,4,128,256),BNorm(256))

function pool_features(x)
    pooled_features = sum(x, dims=[1,2])
end

obj_detector = Chain(Dense(256,58))
