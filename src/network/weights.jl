using Random

function xavier_init(input_dim, output_dim)
    stddev = sqrt(2.0 / (input_dim + output_dim))
    return stddev * randn(output_dim, input_dim)
end

function init_weights()
    rnn_weights = Variable(xavier_init(196, 64), name = "rnn_weights")
    rnn_recurrent_weights = Variable(xavier_init(64, 64), name = "rnn_recurrent_weights")
    rnn_bias = Variable(zeros(64), name = "rnn_bias")
    
    dense_weights = Variable(xavier_init(64, 10), name = "dense_weights")
    dense_bias = Variable(zeros(10), name = "dense_bias")

    return rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias
end