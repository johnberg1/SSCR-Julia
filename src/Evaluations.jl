# This file implements the loss functions
# For now it only includes the generator loss
using Knet
using Statistics

function HingeAdversarialGenerator(fake)
    return mean(-1.*fake)
end

function AdversarialGenerator(fake)
    return mean(softplus(-1.*fake))
end

function softplus(x;beta=1)
    1/beta * log.(1 + exp.(beta.*x))
end
