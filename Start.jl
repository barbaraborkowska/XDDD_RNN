include("./src/network/rnn_custom.jl")
include("./src/network/rnn.jl")

using MLDatasets: MNIST
using Flux

train_data = MNIST(:train)
test_data = MNIST(:test)

x_train = reshape(train_data.features, 28 * 28, :)
y_train = Flux.onehotbatch(train_data.targets, 0:9)
x_test = reshape(test_data.features, 28 * 28, :)
y_test = Flux.onehotbatch(test_data.targets, 0:9)

r = RNN_CUST(5, 100, 15e-3, [[rnn, tanh], [dense, identity]])


using JET
# @profview train(r, x_train, y_train)
# @report_opt train(r, x_train, y_train)
# @code_warntype train(r, x_train, y_train)

train(r, x_train, y_train)


test(r,x_test, y_test)

