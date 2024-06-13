include("./graph/nodes.jl")
import Base: tanh

dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = w * x + b
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = tuple(w' * g, g * x', g)

tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
backward(::BroadcastedOperator{typeof(tanh)}, x, g) = return tuple(g .* (1 .- tanh.(x).^2))

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)

function rnn(x::GraphNode, w::GraphNode, w_rec::GraphNode, b::GraphNode, h_prev::GraphNode)
    return BroadcastedOperator(rnn, x, w, w_rec, b, h_prev)
end

function forward(::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev)
    return (w * x .+ w_rec * h_prev .+ b)
end
function backward(op::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev, g)
    h = forward(op, x, w, w_rec, b, h_prev)  # Calculate current hidden state
    grad_h = g .* (1 .- h.^2)  # Gradient through the tanh non-linearity
    grad_x = w' * grad_h  # Gradient w.r.t. input
    grad_w = grad_h * x'  # Gradient w.r.t. input weights
    grad_w_rec = grad_h * h_prev'  # Gradient w.r.t. recurrent weights
    grad_b = grad_h  # Gradient w.r.t. bias
    grad_h_prev = w_rec' * grad_h  # Gradient w.r.t. previous hidden state
    return (grad_x, grad_w, grad_w_rec, grad_b, grad_h_prev)
end
