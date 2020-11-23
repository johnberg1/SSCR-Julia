using Knet
using CUDA
include("Layers.jl")

moments1 = [bnmoments(), bnmoments()]
params1 = [param(bnparams(1024)), param(bnparams(1024))]
ResUpBlock1 = Chain(BNorm(1024,moments1[1],params1[1],true),
                   Upsample(2),
                   Conv3x3(1024,1024,false),
                   BNorm(1024,moments1[2],params1[2],true),
                   Conv3x3(1024,1024,false))
Residual1 = Chain(Upsample(2),Conv1x1(1024,1024,false))

moments2 = [bnmoments(), bnmoments()]
params2 = [param(bnparams(1024)), param(bnparams(512))]
ResUpBlock2 = Chain(BNorm(1024,moments2[1],params2[1],true),
                   Upsample(2),
                   Conv3x3(1024,512,false),
                   BNorm(512,moments2[2],params2[2],true),
                   Conv3x3(512,512,false))
Residual2 = Chain(Upsample(2),Conv1x1(1024,512,false))

moments3 = [bnmoments(), bnmoments()]
params3 = [param(bnparams(1024)), param(bnparams(256))]
ResUpBlock3 = Chain(BNorm(1024,moments3[1],params3[1],true),
                   Upsample(2),
                   Conv3x3(1024,256,false),
                   BNorm(256,moments3[2],params3[2],true),
                   Conv3x3(256,256,false))
Residual3 = Chain(Upsample(2),Conv1x1(1024,256,false))

moments4 = [bnmoments(), bnmoments()]
params4 = [param(bnparams(256)), param(bnparams(128))]
ResUpBlock4 = Chain(BNorm(256,moments4[1],params4[1],true),
                   Upsample(2),
                   Conv3x3(256,128,false),
                   BNorm(128,moments4[2],params4[2],true),
                   Conv3x3(128,128,false))
Residual4 = Chain(Upsample(2),Conv1x1(256,128,false))

moments5 = [bnmoments(), bnmoments()]
params5 = [param(bnparams(128)), param(bnparams(64))]
ResUpBlock5 = Chain(BNorm(128,moments5[1],params5[1],true),
                   Upsample(2),
                   Conv3x3(128,64,false),
                   BNorm(64,moments5[2],params5[2],true),
                   Conv3x3(64,64,false))
Residual5 = Chain(Upsample(2),Conv1x1(128,64,false))

conditioning_dim = 128
rnn_out_dim = 1024
noise_dim = 100
batch_size = 5
image_feat_dim = 512
ConditionProjector = Chain(Dense(rnn_out_dim,conditioning_dim))
fc1 = Chain(Dense(noise_dim + conditioning_dim,1024 * 4 * 4))
params = param(bnparams(64))
moments = bnmoments()
bn = BNorm(64, moments, params,true)
conv = Chain(Conv3x3(64,3,false))

function forward(y,z,img_feats)
    y_cond = ConditionProjector(y)
    z = vcat(z, y_cond)
    x = fc1(z)
    x = reshape(x, 4, 4, 1024, batch_size)

    x_res = Residual1(x)
    x = ResUpBlock1(x)
    x = x + x_res

    x_res = Residual2(x)
    x = ResUpBlock2(x)
    x = x + x_res

    x = cat(x, img_feats, dims=3)

    x_res = Residual3(x)
    x = ResUpBlock3(x)
    x = x + x_res

    x_res = Residual4(x)
    x = ResUpBlock4(x)
    x = x + x_res

    x_res = Residual5(x)
    x = ResUpBlock5(x)
    x = x + x_res

    x = bn(x)
    x = conv(x)
    x = tanh.(x)
end
