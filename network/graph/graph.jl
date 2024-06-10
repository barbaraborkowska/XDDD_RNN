function build_graph(train_x1, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias)
	l1 = dense(train_x1, rnn_weights, rnn_bias) |> relu
	l2 = dense(l1, dense_weights, dense_bias) |> identity

    # Recurent CELL TODO

	e = cross_entropy_loss(l2, train_y)

	return topological_sort(e)
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :gradient)
			node.batch_gradient ./= batch_size
            node.output -= lr * node.batch_gradient 
            node.batch_gradient .= 0
        end
    end
end