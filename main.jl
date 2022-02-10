using LinearAlgebra, BenchmarkTools, SparseArrays, Random

Nη, Nζ = 100, 99
μ = 0.0
ne = 10000


vη = vζ = [-1, 0, +1]  # allowed variable values
η, ζ = rand(vals, Nη), rand(vals, Nζ)
# η[1] = 0
nηs, nζs = 1:Nη, 1:Nζ
Λ2 = ones(Nζ, Nζ)  # lower △ matrix with zeros along the first row and column (set later...)
Λ1 = ones(Nζ)'
Λ1[1] = 0
X2 = ones(Nζ, Nη)  # lower △ matrix with zeros on the 0,-1 diagonals (set later...)
X1 = ones(Nζ)'
X1[1] = 0
Xμ = ones(Nζ)'

function ΦSB(ζ::Array{Int64, 1}, η::Array{Int64, 1}, Λ1::Adjoint{Float64, Vector{Float64}}, Λ2::Matrix{Float64}, Xμ::Adjoint{Float64, Vector{Float64}}, X1::Adjoint{Float64, Vector{Float64}}, X2::Matrix{Float64}, μ::Float64) 
    ζ'*Λ2*ζ + ζ[1]*Λ1*ζ + ζ'*X2*η + 1im*η[1]*X1*ζ + 1im*μ*Xμ*ζ
end

nacc = 0
nrej = 0
for e=1:ne
    for i in shuffle(nζs), j in shuffle(nηs)
        ζ0, η0 = ζ[i], η[j]  # save old config
        ΦSB0 = ΦSB(ζ, η, Λ1, Λ2, Xμ, X1, X2, μ)  # compute old action
        ζ[i], η[j] = rand(vζ), rand(vη)  # attempt update
        ΦSB1 = ΦSB(ζ, η, Λ1, Λ2, Xμ, X1, X2, μ)  # compute updated action
        ΔΦSB = ΦSB1 - ΦSB0  # action difference
        # @show exp(-ΔΦSB)
        if rand() < real(exp(-ΔΦSB))  # MC
            nacc += 1
        else
            nrej += 1
            ζ[i], η[j] = ζ0, η0  # restore old config
        end
    end
    @info e nacc/(nacc+nrej)
end
