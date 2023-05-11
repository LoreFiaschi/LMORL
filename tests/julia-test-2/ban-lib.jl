abstract type AbstractAlgNum <: Number end
const SIZE = 3 ;
# Ban declaration
mutable struct Ban <: AbstractAlgNum
# Members
    p :: Int
    num :: Array{T, 1 } where T<: Real
# Constructor
    Ban( p :: Int , num :: Array{T, 1 } , check :: Bool ) where T <: Real = new( p , copy(num ) )
    Ban( p :: Int , num :: Array{T, 1 } ) where T <: Real = ( _constraints_satisfaction( p , num ) && new( p , copy(num ) ) )
    Ban( a :: Ban ) = new( a.p , copy(a.num) )
    Ban( x :: Bool ) = one( Ban )
    Ban( x :: T) where T<: Real = ifelse( isinf(x) , Ban( 0 , ones(SIZE).*x ) , one(Ban)*x )
end






# α constant
const α = Ban(  1, [ one( Int64 ) ; zeros( Int64 , SIZE −1 ) ] , false );
# η constant
const η = Ban( −1, [ one( Int64 ) ; zeros( Int64 , SIZE −1 ) ] , false );

println("value of α: ", α)

println("value of η: ", η)

rand_ban_true_1 = Ban(true)
rand_ban_true_2 = Ban(true)

rand_ban_false_1 = Ban(false)

println("rand_ban_true_1: ", rand_ban_true_1)
println("rand_ban_true_2: ", rand_ban_true_2)

println("rand_ban_false_1: ", rand_ban_false_1)

#println("value of α + η: ", α + η)

