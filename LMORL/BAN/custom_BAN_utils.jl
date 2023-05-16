# you must include this file after including 
# a BAN library

function parse_ban_from_array(reward_list, ban_size)
    # we convert the first ban_size components of the reward 
    # to a BAN, 
    # if there are more than ban_size components in the reward, the
    # latter ones are ignored
    ## NOTE: similar code at ref 4571
    ban_reward = Ban(0, reward_list[1:ban_size])
    #
    if reward_list[1] == 0:
        # TODO: should we check if the reward is made only of zeros?
        to_normal_form!(ban_reward)
    return ban_reward