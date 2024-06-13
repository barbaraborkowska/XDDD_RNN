include("./graph/topo_sort.jl")
include("./weights.jl")
include("./train/backpropagation.jl")
include("./train/forward.jl")
include("./graph/graph.jl")
include("./cross_entropy_loss.jl")

function train(x, y, epochs, batch_size, learning_rate)
    rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias = init_weights()
   
    global num_of_correct_clasiffications = 0
    global num_of_clasiffications = 0

    for epoch in 1:epochs
        epoch_loss = 0.0
        num_of_samples = size(x, 2)

        for j in 1:num_of_samples
            train_x = x[:,j]
            train_y = Constant(y[:,j])
            graph = build_graph(train_x, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias);
                
            epoch_loss += forward!(graph)
            backward!(graph)

            if j % batch_size == 0
				update_weights!(graph, learning_rate, batch_size)
			end
        end

        println("Epoch: ", epoch,".  AVG loss: ", epoch_loss  / num_of_samples)
		println("Accuracy: ", num_of_correct_clasiffications/num_of_clasiffications, " (recognized ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
    end

    return rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias
end


function test(x, y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias)
	num_of_samples = size(x, 2)

	global num_of_correct_clasiffications = 0
	global num_of_clasiffications = 0

	for j=1:num_of_samples
		test_x = x[:,j]
        test_y = Constant(y[:,j])

        graph = build_graph(test_x, test_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias);

		forward!(graph)
	end

	println("\n")
	println("Test accuracy: ", num_of_correct_clasiffications/num_of_clasiffications, " (recognized ", num_of_correct_clasiffications, "/", num_of_clasiffications, ")\n")
end