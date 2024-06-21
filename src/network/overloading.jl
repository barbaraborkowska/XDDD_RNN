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

# Optimized backward function
function backward(op::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev, g)
    h = forward(op, x, w, w_rec, b, h_prev)  # OPTIMIZATION
    grad_h = g .* (1 .- h.^2)

    grad_h = g .* (1 .- h.^2)

    grad_x = w' * grad_h
    grad_w = grad_h * x'
    grad_w_rec = grad_h * h_prev'
    grad_b = sum(grad_h, dims=2)
    grad_h_prev = w_rec' * grad_h
   

    return (grad_x, grad_w, grad_w_rec, grad_b, grad_h_prev)
end
