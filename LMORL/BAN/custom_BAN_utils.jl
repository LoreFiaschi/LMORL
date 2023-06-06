# you must include this file after including 
# a BAN library

function parse_ban_from_array(reward_list, ban_size)
    # we convert the first ban_size components of the reward 
    # to a BAN, 
    # if there are more than ban_size components in the reward, the
    # latter ones are ignored
    
    #ban_reward = Ban(0, reward_list[1:ban_size], false)
    ##
    #if reward_list[1] == 0
    #    # TODO: should we check if the reward is made only of zeros?
    #    _to_normal_form!(ban_reward)
    #end
    #return ban_reward

    ##ALTERNATIVE VERSION, similar code at ref 4571
    p = 0
    while reward_list[1]==0
        reward_list= circshift(reward_list, -1)
        p -= 1
        if p <= -(ban_size)
            #prevent infinite loop in case of zeros reward
            p = 0
            break
        end
    end
    reward_ban = Ban(p, reward_list, false)
    return reward_ban

end

# overloading of add_experience_custom_types! to manage different state Array types (Int, Float, ...)

function add_experience_custom_types!(agent::DQNAgent, state::Array{Float32,1},action_index::Int64,reward_list::Array{Float64,1},next_state::Array{Float32,1},done::Bool)
    add_experience!(agent,state,convert(Int32, action_index), parse_ban_from_array(reward_list, BAN_SIZE), next_state, done)
end

function add_experience_custom_types!(agent::DQNAgent, state::Array{Float32,1},action_index::Int64,reward_list::Array{Float32,1},next_state::Array{Float32,1},done::Bool)
    add_experience!(agent,state,convert(Int32, action_index), parse_ban_from_array(reward_list, BAN_SIZE), next_state, done)
end

function add_experience_custom_types!(agent::DQNAgent, state::Array{Int32,1},action_index::Int64,reward_list::Array{Float32,1},next_state::Array{Int32,1},done::Bool)
    add_experience!(agent,state,convert(Int32, action_index), parse_ban_from_array(reward_list, BAN_SIZE), next_state, done)
end

function add_experience_custom_types!(agent::DQNAgent, state::StateType,action_index::Int64,reward_list::Array{Float32,1},next_state::StateType,done::Bool)
    add_experience!(agent,state,convert(Int32, action_index), parse_ban_from_array(reward_list, BAN_SIZE), next_state, done)
end

#function call_plot_from_python(num_episodes::Int, rewards_list_of_list::Array{Array{Float64,1}, 1}, rlplot::Bool, title::String, call_plot::Bool)
function call_plot_from_python(num_episodes::Int, rewards_matrix::Matrix{Float64}, rlplot::Bool, title::String, call_plot::Bool)
    x=convert(Vector{Ban}, range(1,num_episodes))
    y=Ban[] #Vector{Ban}#(undef,0)
    for i = 1:size(rewards_matrix,1)
        append!(y, parse_ban_from_array(rewards_matrix[i, :], BAN_SIZE)) 
    end
    #for reward_list in rewards_list_of_list
    #    push!(y, parse_ban_from_array(reward_list, BAN_SIZE)) 
    #end
    plot(x, y, rlplot=rlplot, title=title)
    if call_plot
        PyPlot.show()
    end
    #x::Vector{Ban}, y::Vector{Ban}; kwargs...
end