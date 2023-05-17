# packages "Format","PyPlot","Distributions",
# "Parameters","Flux","ChainRulesCore","ProgressBars","Gym","BSON","Zygote"
# must be installed in order to use this script

if BAN_SIZE == 3
    include("../BAN_s3_isbits.jl")
elseif BAN_SIZE == 2
    include("../BAN.jl")
else
    SIZE = BAN_SIZE
    include("../BAN.jl")
end

include("../BanPlots.jl")

using Distributions
using Parameters
using Random
using .BAN
using Flux
using ChainRulesCore
using ProgressBars
using LinearAlgebra
using Gym


Random.seed!(777)
rng=MersenneTwister(777)

Ban_glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Ban, dims...) .- (0.5*Ban(0,ones(BAN_SIZE),false))) .* sqrt(24.0*Ban(0,ones(BAN_SIZE),false) / sum(Flux.nfan(dims...)))
Ban_glorot_uniform(rng::AbstractRNG) = (dims...) -> Ban_glorot_uniform(rng, dims...)

function banStep(env, action)
    state, r, done, information = step!(env, action)
    i=0
    #(ref 4571)
    while r[1]==0
        r=circshift(r,-1)
        i-=1
    end
    # end 4571
    # modified banStep so that it works even
    # when the reward size is higher than the BAN SIZE
    reward=Ban(i,r[1:BAN_SIZE],false)
    return state,reward,done,information
end

StateType=Union{Array{Float32,1},Array{Int32,1}}


@with_kw mutable struct Replay
    state::StateType
    action::Int32
    reward::Ban
    next_state::StateType
    done::Bool
end




@with_kw mutable struct DQNAgent
    epsilon::Float32 = 1
    epsilon_decay::Float32 = 0.99
    epsilon_min::Float32 = 0.0
    𝜏::Float32=0.001
    gamma::Float32 = 0.99
    learning_rate::Float32 = 0.1
    numactions::Int32 
    actionspace::Array{Int32,1}
    #previous_state::StateType=zeros(2)
    action::Int32=0
    memory::Vector{Replay}=Vector{Replay}(undef,0)
    input_size::Int32=2
    max_memory::Int32=100
    occupied_memory::Int32=0
    last_inserted_index::Int32=0
    hidden_size::Int32=10
    train_start::Int32=100
    input_layer=Dense(input_size,hidden_size,relu;init=Flux.glorot_uniform(rng))
    output_layer=Dense(hidden_size,numactions;init=Ban_glorot_uniform(rng))
    model=f64(Chain(input_layer,Dense(hidden_size,hidden_size,relu;init=Flux.glorot_uniform(rng)),output_layer))
    target_model=Chain(Dense(input_size,hidden_size,relu),Dense(hidden_size,hidden_size,relu),Dense(hidden_size,numactions))
    batch_size::Int64=64;
    
end

function act!(agent::DQNAgent,observation::StateType)
    if rand(Uniform())>agent.epsilon
        q_value=agent.model(observation)
        agent.action=argmax(q_value)
    else
        agent.action=rand(1:agent.numactions)
    end
    return agent.action
end

#IMPLEMENTATION WITH FIXED MEMORY POSITION ARRAY
function add_experience!(agent::DQNAgent,state::StateType,action::Int32,reward::Ban,next_state::StateType,done::Bool)
    
    if agent.occupied_memory < agent.max_memory
        r=Replay(state,action,reward,next_state,done)
        push!(agent.memory, r)
        agent.occupied_memory+=1
        agent.last_inserted_index += 1
    else
        agent.last_inserted_index=( (agent.last_inserted_index ) % 100 ) + 1
        agent.memory[agent.last_inserted_index].state = state
        agent.memory[agent.last_inserted_index].action = action
        agent.memory[agent.last_inserted_index].reward = reward
        agent.memory[agent.last_inserted_index].next_state = next_state
        agent.memory[agent.last_inserted_index].done = done
          
    end
    
end

# OLD IMPLEMENTATION:
#function add_experience!(agent::DQNAgent,state::Array{Float32,1},action::Int32,reward::Ban,next_state::Array{Float32,1},done::Bool)
#    r=Replay(state,action,reward,next_state,done)
#    if agent.occupied_memory >= agent.max_memory #length(agent.memory)
#        popfirst!(agent.memory) #deleteat!(agent.memory, 1)
#        agent.occupied_memory -= 1
#    end
#    push!(agent.memory, r)
#    agent.occupied_memory+=1
#end


##TOTEST: a method to store a vector of timesteps in batch
#function add_experience_batch!(agent::DQNAgent,state::Array{Float32,2},action::Array{Int32,1},reward::Array{Ban,1},next_state::Array{Float32,2},done::Array{Bool,1}, how_many::Int64)
#    for index in 1:how_many
#        add_experience!(agent, state[index], action[index], reward[index], next_state[index], done[index])
#    end
#end

function get_target_q_value(agent::DQNAgent,next_state::StateType)
    max_q_value= maximum(agent.target_model(next_state))
    return max_q_value
end


function update_target_model!(agent::DQNAgent)
    agent.target_model=deepcopy(agent.model)
end


function update_epsilon!(agent::DQNAgent)
        if agent.occupied_memory < agent.train_start#length(agent.memory) < agent.train_start
            return
        end
        if agent.epsilon > agent.epsilon_min
            agent.epsilon *= agent.epsilon_decay
        end
end

function ChainRulesCore.rrule(::typeof((*)),A::AbstractMatrix, x::AbstractVector)
    y = A*x
    function ban_matvec_pullback(ȳ)
            f̄=NoTangent()
            ā =ȳ*x'
            b̄ =A'*ȳ     
        return f̄, ā, b̄
    end
    return y, ban_matvec_pullback
end

function ChainRulesCore.rrule(::typeof((*)),A::AbstractMatrix, B::AbstractMatrix)
    y = A*B
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function ban_matmat_pullback(ȳ)
            f̄=NoTangent()
            ā =ȳ*B'
            b̄ =A'*ȳ      
        return f̄, project_A(ā), project_B(b̄)
    end
    return y, ban_matmat_pullback
end

Flux.@adjoint abs2(x::Ban) = abs2(x), Δ -> (Ban(Δ)*(x + x),)


function experience_replay!(agent::DQNAgent)
    if length(agent.memory) < agent.train_start
        return
    end
    loss(x, y) = Flux.Losses.mse(agent.model(x), y)
    ps = Flux.params(agent.model)
    opt = ADAM(agent.learning_rate)
    mini_batch=Distributions.sample(agent.memory,agent.batch_size)
    state_batch=Array{Float32,2}(undef, agent.input_size, 0)
    q_values_batch=Array{Ban,2}(undef, agent.numactions, 0)
    for r in mini_batch
         q_value = agent.model(r.state)
        if r.done
            q_value[r.action] = r.reward
        else
            target_q_value = get_target_q_value(agent,r.next_state)
            q_value[r.action] = r.reward +  target_q_value*agent.gamma
        end
        state_batch=hcat(state_batch,r.state)
        q_values_batch=hcat(q_values_batch,q_value)
    end
    data = Flux.DataLoader((state_batch, q_values_batch), batchsize=agent.batch_size) 
    ps = Flux.params(agent.model)
    opt = ADAM(agent.learning_rate)
    for d in data
        gs = gradient(ps) do
            loss(d...)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
end

Base.Float64(a::Ban)  =  Float64(BAN.standard_part(a))


using BSON
function hybrid_agent_learning(env, agent, episodes,mname,reward_threshold)
    rewards=Ban[]
    avgrewards=Ban[]
    timings=zeros(0)
    solved=false
    i=1
    while i<=episodes && !solved
        state, information=reset!(env)
        done=false
        t=0
        totrew=0
        while !done
            tmng=@elapsed begin
                action=act!(agent,state)
                next_state,reward,done=banStep(env,agent.actionspace[action])
                totrew+=reward
                add_experience!(agent,state,action,reward,next_state,done)
                state=next_state
                t+=1
                if t%replay_frequency==0
                    experience_replay!(agent)
                end
            end
            append!(timings,tmng)
        end
        update_target_model!(agent)
        update_epsilon!(agent)
        append!(rewards,totrew)
        if i%50==0  #50 aka dump_period
            weights=Flux.params(agent.model)
            BSON.@save mname weights
        end
        if i>=100
            avg_reward=mean(rewards[i-99:i])
            if (avg_reward.num1+avg_reward.num2+avg_reward.num3) >= reward_threshold
                solved= true
                println("Solved in $i episodes.")
    
                weights=Flux.params(agent.model)
                BSON.@save mname weights
            end
        else 
            avg_reward=mean(rewards[1:i])
        end
        append!(avgrewards,avg_reward)
        println("Episode $i , reward = $totrew 100AvgReward= $avg_reward")
        i+=1
    end
    return rewards, avgrewards, timings
end