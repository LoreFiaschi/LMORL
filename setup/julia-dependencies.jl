using Pkg;
for p in ( "Format","PyPlot","Distributions","Parameters","Flux","ChainRulesCore","ProgressBars","Gym","BSON","Zygote")
    if (Pkg.status(p) == nothing)
         Pkg.add(p)
    end
end