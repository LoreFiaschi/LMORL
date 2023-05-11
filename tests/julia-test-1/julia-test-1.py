from julia.api import Julia

#import julia
#julia.install()

from julia import Main


jl = Julia(compiled_modules=False)


data = "fenomitaly"

jl.eval('include("julia-test-1.jl")')
Main.data = data
ret_param = jl.eval("print(data)")


print(f"value returned by Julia script: ", ret_param)