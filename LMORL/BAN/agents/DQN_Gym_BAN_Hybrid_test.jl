BAN_SIZE=3

include("DQN_Gym_BAN_Hybrid.jl")

########################################
############## SETUP ###################
########################################

using BSON
using Zygote
Random.seed!(777)
Random.seed!(rng,777)
env = GymEnv("LunarLander-v2-mo")
st, foo = reset!(env)
inputsize=length(st)
#print(inputsize)
episodes=10
actions=[0,1,2,3]
reward_threshold=200
replay_frequency=1
agent=DQNAgent(input_size=inputsize,numactions=4,actionspace=actions,max_memory=100000,learning_rate= 0.0001,epsilon_decay = 0.995,epsilon_min=0.1, batch_size=64,train_start=64,hidden_size=128)
solved=false
mname="TESTTIMEHYBDQN.bson"
agent.target_model=deepcopy(agent.model)
reward_threshold=200
rewards,avgrewards,timings= hybrid_learning(env, agent, episodes, mname, reward_threshold)
################################
x=convert(Vector{Ban}, range(1,episodes))
println(x)
println(rewards)
plot(x,rewards,rlplot=true, figtitle="Foo")
###
x=convert(Vector{Ban}, range(1,episodes))
plot(x,avgrewards,rlplot=true)
PyPlot.show()
##################################
