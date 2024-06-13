include("./graph/topo_sort.jl")
include("./weights.jl")
include("./train/backpropagation.jl")
include("./train/forward.jl")
include("./graph/graph.jl")
include("./cross_entropy_loss.jl")
include("./rnn_custom.jl")


function train(r::RNN_CUST, x::Any, y::Any)

    println("TEST RUN WITHOUT LEARNING ---- SHOULD BE AROUND 10% ACCURACY")
    test(r, x, y)
    println("------------------------------------------------------")

    global good_clasiff = 0
    global all_clasiff = 0

    @time for epoch in 1:r.epochs
        epoch_loss = 0.0
        num_of_samples = size(x, 2)

        for j in 1:num_of_samples
            train_x = x[:,j]
            train_y = Constant(y[:,j])
            graph = build_graph(train_x, train_y, r.rnn_weights, r.rnn_recurrent_weights, r.rnn_bias, r.dense_weights, r.dense_bias, r.arch);
                
            epoch_loss += forward!(graph)
            backward!(graph)

            if j % r.batch_size == 0
				update_weights!(graph, r.learning_rate, r.batch_size)
			end
        end

        println("EPOCH: ", epoch,".  AVG LOSS: ", epoch_loss  / num_of_samples)
		println("ACCURACY: ", good_clasiff/all_clasiff, " (RECOGNIZED ", good_clasiff, "/", all_clasiff, ")\n")
    end
end


function test(r::RNN_CUST,x, y)
	num_of_samples = size(x, 2)

	global good_clasiff = 0
	global all_clasiff = 0

	for j=1:num_of_samples
		test_x = x[:,j]
        test_y = Constant(y[:,j])

        graph = build_graph(test_x, test_y, r.rnn_weights, r.rnn_recurrent_weights, r.rnn_bias, r.dense_weights, r.dense_bias, r.arch);

		forward!(graph)
	end

    println("TEST ACCURACY: ", good_clasiff/all_clasiff, " (RECOGNIZED ", good_clasiff, "/", all_clasiff, ")")
end