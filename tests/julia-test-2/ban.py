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
    
    def print(self):
        import string

        superscript_map = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
            "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺","-": "⁻"}
        trans = str.maketrans(
            ''.join(superscript_map.keys()),
            ''.join(superscript_map.values()))
        
        char = "α"

        for index, el in enumerate(self.num):
            exp = self.p - index
            exp_str = str(exp).translate(trans)
            if index > 0 and el >= 0: print(" + ", end="")
            print(f"{el}{char}{exp_str}", end="")

        print()
        