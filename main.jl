using LinearAlgebra, Random, QuadGK, PyPlot

I(ω::Float64; α::Float64=1.0, ωc::Float64=1.0) = 2π*α*ω*exp(-ω/ωc)
function Q(t::Float64; ω∞::Float64=10.0, α::Float64=1.0, β::Float64=1.0, ωc::Float64=1.0)
    return quadgk(ω -> 1/π*I(ω; α=α, ωc=ωc)/ω^2*(coth(ω*β/2)*(1-cos(ω*t)) + 1im*sin(ω*t)), 0.0, ω∞)[1]
end

# let
    μ = 1.0
    ne = 20
    q = 100
    τ = 10.0
    vη = vζ = [-1, 0, +1]  # allowed variable values
    Nη = Nζ = q
    η, ζ = rand(vη, Nη), rand(vζ, Nζ)

    # helper functions to set matrix elements
    d²Qdτ²(j) = Q((j+1)*τ) + Q((j-1)*τ) - 2*Q(j*τ)
    d²Qdτ²oh(j) = Q((j-0.5)*τ) + Q((j-1.5)*τ) - 2*Q((j-1)*τ) + Q((j-2)*τ)

    nηs, nζs = 1:Nη, 1:Nζ
    Λ2 = zeros(ComplexF64, Nζ, Nζ)  # lower △ matrix with zeros along the diagonal
    X2 = zeros(Nζ, Nη)  # lower △ matrix with zeros along the 0,-1 diagonals
    Λ1 = zeros(Nζ)'
    X1 = zeros(Nζ)'
    Xμ = zeros(Nζ)'

    Λ2[2,2] = Q(τ)
    Λ1[1] = real(Q(τ/2))
    Xμ[1] = -imag(Q(τ/2))
    for x in 2:size(Λ2, 1)
        for y in 1:x # 1:size(Λ2, 2)
            # @show x, y
            j = x-y
            if 1 <= j <= q-2
                tp = d²Qdτ²(j)
                Λ2[x, y] = y>1 ? real(tp) : 0
                X2[x, y] = y<x ? imag(tp) : 0
            end
            if j==0
                Λ2[x, y] = Λ2[2,2]
            end
        end 
        tp = d²Qdτ²oh(x)
        Λ1[x] = real(tp)
        X1[x] = imag(tp)
        Xμ[x] = -imag(tp)
    end

    function ΦSB(ζ::Array{Int64, 1}, η::Array{Int64, 1}, Λ1::Adjoint{Float64, Vector{Float64}}, Λ2::Matrix{ComplexF64}, Xμ::Adjoint{Float64, Vector{Float64}}, X1::Adjoint{Float64, Vector{Float64}}, X2::Matrix{Float64}, μ::Float64) 
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
        @show e, accr
    end

    fig, ax = plt.subplots()
    ax.plot(η, label="η")
    ax.plot(ζ, label="ζ")
    ax.set_xlabel("j")
    ax.legend()
    fig.savefig("cfg.png")
    plt.close()
# end