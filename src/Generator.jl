import Knet
using Knet: conv4, pool, mat, KnetArray, zeroone, progress, sgd, param, param0, dropout, relu, batchnorm, bnmoments, bnparams

# Not complete
struct Conv3x3; w; f; end
(c::Conv3x3)(x) = c.f.(conv4(c.w, x, padding = 0, stride = 1))
Conv3x3(cx::Int,cy::Int,f=relu) = Conv3x3(param(3,3,cx,cy),f)

struct Conv1x1; w; f; end
(c::Conv1x1)(x) = c.f.(conv4(c.w, x, padding = 1, stride = 1))
Conv1x1(cx::Int,cy::Int,f=relu) = Conv1x1(param(1,1,cx,cy),f)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

struct BNorm; C; end
(bn::BNorm)(x) = batchnorm(x ,bnmoments(),param(bnparams(bn.C)))
