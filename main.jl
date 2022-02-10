using LinearAlgebra, BenchmarkTools, SparseArrays, Random, QuadGK

I(ω::Float64; α::Float64=1.0, ωc::Float64=1.0) = 2π*α*ω*exp(-ω/ωc)
function Q(t::Float64, ω∞::Float64=10.0; α::Float64=1.0, β::Float64=1.0, ωc::Float64=1.0)
    return quadgk(ω -> 1/π*I(ω; α=α, ωc=ωc)/ω^2*(coth(ω*β/2)*(1-cos(ω*t)) + 1im*sin(ω*t)), 0.0, ω∞)[1]
end

let
    μ = 0.0
    ne = 100
    q = 10
    τ = 20.0
    vη = vζ = [-1, 0, +1]  # allowed variable values
    Nη = Nζ = q
    η, ζ = rand(vη, Nη), rand(vζ, Nζ)

    Λ2₀ = real(Q(τ))
    d²τQ(j) = ((j+1)*τ) + Q((j-1)*τ) - 2*Q(j*τ)
    Λ2_1lejleqm2(j::Int64) = real(d²Qτ(j))
    X2_1lejleqm2(j::Int64) = imag(Q((j+1)*τ) + Q((j-1)*τ) - 2*Q(j*τ))
    Λ1₁ = real(Q(τ/2))
    Λ1_2lejleq(j::Int64) = real(Q((j-1/2)*τ) - Q((j-1)*τ) - Q((j-3/2)*τ) - Q((j-2)*τ))
    X1_2lejleq(j::Int64) = real(Q((j-1/2)*τ) - Q((j-1)*τ) - Q((j-3/2)*τ) - Q((j-2)*τ))
    Xμ₁ = -imag(Q(τ/2))
    Xμ₁2lejleq(j::Int64) = -imag(Q((j-1/2)*τ) - Q((j-3/2)*τ))

    nηs, nζs = 1:Nη, 1:Nζ
    Λ2 = zeros(Nζ, Nζ)  # lower △ matrix with zeros along the diagonal
    for x in 1:size(Λ2, 1), y in 1:x-1 # 1:size(Λ2, 2)
        @show x, y
        j = x-y
        if 1 <= j <= q-2
            Λ2[x, y] = d²τ
        end
    end
    Λ1 = zeros(Nζ)'
    Λ1[1] = Λ1₁
    X2 = zeros(Nζ, Nη)  # lower △ matrix with zeros along the 0,-1 diagonals
    for x in 1:size(X2, 1), y in 1:x-1 # 1:size(Λ2, 2)
        @show x, y
        j = x-y
        if 1 <= j <= q-2
            X2[x, y] = X2_1lejleqm2(j)
        end
    end
    X1 = zeros(Nζ)'
    X1[1] = 0
    Xμ = zeros(Nζ)'

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
        accr = nacc/(nacc+nrej)
        @show e,accr
    end
end