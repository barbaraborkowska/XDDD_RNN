include("./graph/topo_sort.jl")
include("./weights.jl")
include("./train/backpropagation.jl")
include("./train/forward.jl")
include("./graph/graph.jl")
include("./cross_entropy_loss.jl")
include("./rnn_custom.jl")
using Plots


function train(r::RNN_CUST, x::Any, y::Any)


    
   # println("TEST RUN WITHOUT LEARNING ---- SHOULD BE AROUND 10% ACCURACY")
    #test(r, x, y)
    #println("------------------------------------------------------")

    global good_clasiff = 0
    global all_clasiff = 0
    global_epoch_loss = Vector{Float64}()
    global_accuracy = Vector{Float64}()

    train_y = Constant(y[:,1])
    x_t1 = Variable(x[1:196, 1])
    x_t2 = Variable(x[197:392, 1]) 
    x_t3 = Variable(x[393:588, 1]) 
    x_t4 = Variable(x[589:end, 1]) 

    graph = build_graph(x_t1, x_t2, x_t3, x_t4, train_y, r.rnn_weights, r.rnn_recurrent_weights, r.rnn_bias, r.dense_weights, r.dense_bias, r.arch);

    @time for epoch in 1:r.epochs
        epoch_loss = 0.0
        iter_loss = 0.0
        num_of_samples = size(x, 2)

        global good_clasiff = 0
		global all_clasiff = 0

        for j in 2:num_of_samples
            
            loss = forward!(graph)
            epoch_loss += loss
            iter_loss += loss

            backward!(graph)

            if j % r.batch_size == 0
				update_weights!(graph, r.learning_rate, r.batch_size)
                push!(global_epoch_loss, iter_loss / r.batch_size)
                push!(global_accuracy, good_clasiff/all_clasiff)
				iter_loss = 0.0
			end

            x_t1.output = x[1:196, j]
            x_t2.output = x[197:392, j]
            x_t3.output = x[393:588, j] 
            x_t4.output = x[589:end, j]
            train_y.output = y[:,j]
        end

        println("EPOCH: ", epoch,".  AVG LOSS: ", epoch_loss  / num_of_samples)
		println("ACCURACY: ", good_clasiff/all_clasiff, " (RECOGNIZED ", good_clasiff, "/", all_clasiff, ")\n")
    end

    plt = plot(global_epoch_loss, label="Loss", xlabel="Iteration", ylabel="Loss")
    savefig(plt, "./images/loss.png")

    plt = plot(global_accuracy, label="Accuracy", xlabel="Iteration", ylabel="Accuracy")
    savefig(plt, "./images/accuracy.png")

end


function test(r::RNN_CUST,x, y)
	num_of_samples = size(x, 2)

	global good_clasiff = 0
	global all_clasiff = 0

    train_y = Constant(y[:,1])
    x_t1 = Variable(x[1:196, 1])
    x_t2 = Variable(x[197:392, 1]) 
    x_t3 = Variable(x[393:588, 1]) 
    x_t4 = Variable(x[589:end, 1]) 

    graph = build_graph(x_t1, x_t2, x_t3, x_t4, train_y, r.rnn_weights, r.rnn_recurrent_weights, r.rnn_bias, r.dense_weights, r.dense_bias, r.arch);

	for j=2:num_of_samples
		forward!(graph)

        x_t1.output = x[1:196, j]
        x_t2.output = x[197:392, j]
        x_t3.output = x[393:588, j] 
        x_t4.output = x[589:end, j]
        train_y.output = y[:,j]
	end

    println("TEST ACCURACY: ", good_clasiff/all_clasiff, " (RECOGNIZED ", good_clasiff, "/", all_clasiff, ")")
end