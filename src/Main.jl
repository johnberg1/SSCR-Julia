using Knet
include("ImageEncoder.jl")
include("Generator.jl")
# Sentence encoder is not yet complete so we are using a random vector
# I haven't implemented a loop yet since everything is not complete, but the following code implements a forward pass

x = param(rand(128,128,3,5)) # input image batch
imgFeatures = ImageEncoder(x) # 16 x 16 x 512 x 5
pooledFeatures = pool_features(imgFeatures) # 1 x 1 x 512 x 5
objDetections = obj_detector(pooledFeatures) # 58 x 5

y = param(rand(rnn_out_dim,batch_size)) # Encoded sentence, we start with random since it is not fully implemented
z = param(randn(noise_dim,batch_size)) # Noise to input to the the generator

generatedImage = forward(y,z,imgFeatures) # 128 x 128 x 3 x 5
