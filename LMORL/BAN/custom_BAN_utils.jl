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
    end
    reward_ban = Ban(p, reward_list, false)
    return reward_ban

end