from matplotlib import pyplot as plt 
import string

class Ban:
    p = None
    num = []

    def __init__(self, p : int, num : list) -> None:
        #TODO: integrity checks of inputs
        self.p = p
        self.num = num.copy()


    def sum(a : 'Ban', b : 'Ban') -> 'Ban':
        # we need the length of both a and b num fields
        # we also need to check the starting exponent p of both a and b
        # it could be useful to obtain the "lowest" exponent for alpha (defined as p - (len(Ban.num)))
        max_p_a = a.p
        min_p_a = a.p - len(a.num)

        max_p_b = b.p
        min_p_b = b.p - len(b.num)

        res_p = res_max_p = max(max_p_a, max_p_b)
        res_min_p = min(min_p_a, min_p_b)

        res_len_num = res_max_p - res_min_p

        offset = abs(a.p - b.p)

        # lets initialize res_num by the content of Ban.num of the input Ban with the highest p
        if a.p >= b.p:
            res_num = a.num.copy()
            num_from = b.num
        else:
            res_num = b.num.copy()
            num_from = a.num

        # here we want to extend res_num so that its lenght is res_len_num, we want to fill the extending part with zeros
        res_num.extend( [0]*(res_len_num - len(res_num)) )

        for i in range(len(num_from)):
            res_num[offset + i] += num_from[i]

        return Ban(res_p, res_num)

    
    def print(self, print_on_std_out : bool = True):
        trans = Ban.get_mapping()
        
        char = "α"

        output = ""

        for index, el in enumerate(self.num):
            exp = self.p - index
            exp_str = str(exp).translate(trans)
            if index > 0 and el >= 0: output += " + " #print(" + ", end="")
            output += f"{el}{char}{exp_str}" #print(f"{el}{char}{exp_str}", end="")

        if print_on_std_out:
            print(output)
        return output

    def get_as_ban_string(ban_as_list : list) -> str:
        tmp_ban = Ban(0, ban_as_list)
        return tmp_ban.print(print_on_std_out=False)


    def get_mapping():

        superscript_map = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
            "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺","-": "⁻"}

        trans = str.maketrans(
            ''.join(superscript_map.keys()),
            ''.join(superscript_map.values())) 

        return trans  

    def __display_plot_jl(rewards : list, num_episodes : int, title : str = "", call_plot:bool=True):
        from julia.api import Julia
        from julia import Main
        import os, pathlib

        jl = Julia(compiled_modules=True)

        cur_path = os.getcwd()

        path = pathlib.Path(__file__).parent.resolve()
        path = str(path).replace("\\", "\\\\")
        jl.eval(f"cd(\"{path}\")")

        jl.eval("""
        (@isdefined plot) ? nothing : include(\"../BanPlots.jl\")
        (@isdefined call_plot_from_python) ? nothing : include(\"../custom_BAN_utils.jl\")
        """
        )
        os.chdir(cur_path)

        rlplot = True

        Main.call_plot_from_python(num_episodes, rewards, rlplot, title, call_plot)



    def display_plot(rewards:list, num_episodes:int, title:str = "", call_plot:bool = True, use_BanPlots : bool = False):
            """
            plot the behaviour of the reawards during episodes
            - rewards must be a list of lists, where each element of the parent list is a MO reward and each element 
            of the child list is a component of a reward (considered float)
            - each MO reward is assumed to have the same number of components
            #- call plt.show() to show the generated figure
            - call %matplotlib inline to display inline plot in .ipynb file
            """

            if use_BanPlots:
                if call_plot == False:
                    print(f"setting call_plot to True since cannot return Figure object from Julia")
                    call_plot = True
                Ban.__display_plot_jl(rewards, num_episodes, title, call_plot)
                return
                
            
            tmp=list(zip(*rewards))
            how_many_components = len(tmp)
            
            fig, (ax_list) = plt.subplots(how_many_components, sharex=True)
            fig.subplots_adjust(hspace=0)
            fig.suptitle(title)

            for i in range(how_many_components):
                ax_list[i].set(ylabel='α'+str(-i).translate(Ban.get_mapping()))
                ax_list[i].plot(range(num_episodes), tmp[i])
            

            plt.xlabel("Episodes")
            if call_plot:
                plt.show()
            return fig
    
    def averaged_sequence(sequence : list, window_size : int = 100) -> list:
        """
        - given in input a sequence of vectors, returns the average-smoothed sequence.
        - all the vectors must have the same number of components
        - window_size: the number of vectors to consider for calculating the average
        """
        ret = []
        sequence_len = len(sequence)
        tmp=list(zip(*sequence))
        how_many_components = len(tmp)

        for i in range(sequence_len):
            cur_size = min((window_size, i + 1))
            offset = cur_size 
            tmp_app = []
            for j in range(how_many_components):
                tmp_app.append( sum( tmp[j][i - offset + 1 : i + 1] ) / cur_size )

            ret.append(tmp_app)
        return ret

    def display_execution_time(new_timings:list, legacy_timings:list, title:str = ""):

        #remove the maximum execution time as it is widely larger than others because of Julia object allocation
        for i in range(1):  
            new_timings.remove(max(new_timings))

        xpoints = range(len(legacy_timings)) if (len(new_timings) > len(legacy_timings)) else range(len(new_timings))
        
        ypoints_new = new_timings[:len(xpoints)]
        ypoints_legacy = legacy_timings[:len(xpoints)]
        
        fig = plt.plot(xpoints, ypoints_new)
        plt.plot(xpoints, ypoints_legacy)
        
        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Execution Time")
        plt.legend(['Main Object', 'jl_eval'])

        plt.show()
        return fig
    
