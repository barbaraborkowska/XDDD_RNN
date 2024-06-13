function build_graph(train_x, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias)
    h_prev = Constant(zeros(64))  # Initial hidden state, assuming hidden size is 64

    # Process the input in chunks
    chunk_size = 196
    seq_length = div(784, chunk_size)

    for i in 0:(seq_length-1)
        x_t = Constant(train_x[(i*chunk_size+1):((i+1)*chunk_size)])  # Chunk of input
        h_prev = rnn(x_t, rnn_weights, rnn_recurrent_weights, rnn_bias, h_prev) |> tanh
        @show(rnn_recurrent_weights)
    end
    l2 = dense(h_prev, dense_weights, dense_bias) |> identity
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