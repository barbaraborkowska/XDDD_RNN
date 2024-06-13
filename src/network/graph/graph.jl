function build_graph(train_x, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias, arch)
    h_prev = Constant(zeros(64))  

    seq_length = div(784, 196)

    for i in 0:(seq_length-1)
        x_t = Constant(train_x[(i*196+1):((i+1)*196)]) 
        h_prev = arch[1][1](x_t, rnn_weights, rnn_recurrent_weights, rnn_bias, h_prev) |> arch[1][2]
    end
    
    l2 = arch[2][1](h_prev, dense_weights, dense_bias) |> arch[2][2]
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