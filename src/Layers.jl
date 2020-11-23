using Knet
using CUDA

struct Conv3x3; w; act; f; end
(c::Conv3x3)(x) = if c.act c.f.(conv4(c.w, x, padding = 1, stride = 1)) else conv4(c.w, x, padding = 1, stride = 1) end
Conv3x3(cx::Int,cy::Int,act,f=relu) = Conv3x3(param(3,3,cx,cy),act,f)

struct Conv1x1; w; act; f; end
(c::Conv1x1)(x) = if c.act c.f.(conv4(c.w, x, padding = 0, stride = 1)) else conv4(c.w, x, padding = 0, stride = 1) end
Conv1x1(cx::Int,cy::Int,act,f=relu) = Conv1x1(param(1,1,cx,cy),act,f)

struct Dense; w; b; p; end
(d::Dense)(x) = d.w * mat(dropout(x,d.p)) .+ d.b # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int;pdrop=0) = Dense(param(o,i), param0(o), pdrop)

struct BNorm
    C
    moments
    params
    act
end
(bn::BNorm)(x) = if bn.act relu.(batchnorm(x, bn.moments, bn.params)) else batchnorm(x, bn.moments, bn.params) end

struct Upsample; stride; end
(u::Upsample)(x) = unpool(x, stride=u.stride)

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

struct Conv; w; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, padding = 1, stride = 2))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Conv(param(w1,w2,cx,cy),f)

struct Dense1; w; b; f; p; end
(d::Dense1)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense1(i::Int,o::Int,f=sigm;pdrop=0) = Dense1(param(o,i), param0(o), f, pdrop)
