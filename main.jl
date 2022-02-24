using LinearAlgebra, Random, QuadGK, PyPlot

I(ω::Float64; α::Float64=1.0, ωc::Float64=1.0) = 2π*α*ω*exp(-ω/ωc)
function Q(t::Float64; ω∞::Float64=10.0, α::Float64=1.0, β::Float64=1.0, ωc::Float64=1.0)
    return quadgk(ω -> 1/π*I(ω; α=α, ωc=ωc)/ω^2*(coth(ω*β/2)*(1-cos(ω*t)) + 1im*sin(ω*t)), 0.0, ω∞)[1]
end

# let
    μ = 1.0
    ne = 100
    q = 10
    τ = 0.05
    vη = vζ = [-1, 0, +1]  # allowed variable values
    Nη = Nζ = q
    η, ζ = rand(vη, Nη), rand(vζ, Nζ)

    α = 0.
    β = 0.1
    ωc = 100.

    # helper functions to set matrix elements
    d²Qdτ²(j) = Q((j+1)*τ, α=α, β=β, ωc=ωc) + Q((j-1)*τ, α=α, β=β, ωc=ωc) - 2*Q(j*τ, α=α, β=β, ωc=ωc)
    d²Qdτ²oh(j) = Q((j-0.5)*τ, α=α, β=β, ωc=ωc) + Q((j-1.5)*τ, α=α, β=β, ωc=ωc) - 2*Q((j-1)*τ, α=α, β=β, ωc=ωc) + Q((j-2)*τ, α=α, β=β, ωc=ωc)

    nηs, nζs = 1:Nη, 1:Nζ
    Λ2 = zeros(ComplexF64, Nζ, Nζ)  # lower △ matrix
    X2 = zeros(Nζ, Nη)  # lower △ matrix with zeros along the diagonals
    Λ1 = zeros(Nζ)'
    X1 = zeros(Nζ)'
    Xμ = zeros(Nζ)'

    Λ2[2,2] = Q(τ, α=α, β=β, ωc=ωc)
    Λ1[1] = real(Q(τ/2, α=α, β=β, ωc=ωc))
    Xμ[1] = -imag(Q(τ/2, α=α, β=β, ωc=ωc))
    for j in 2:size(Λ2, 1)
        for k in 1:j # 1:size(Λ2, 2)
            # @show x, y
            j̃ = j-k
            if 1 <= j̃ <= q-2
                tp = d²Qdτ²(j̃)
                Λ2[j, k] = k>1 ? real(tp) : 0
                X2[j, k] = k<j ? imag(tp) : 0
            end
            if j̃==0
                Λ2[j, k] = Λ2[2,2]
            end
        end
        tp = d²Qdτ²oh(j)
        Λ1[j] = real(tp)
        X1[j] = imag(tp)
        Xμ[j] = -imag(tp)
    end

    function ΦSB(ζ::Array{Int64, 1}, η::Array{Int64, 1}, Λ1::Adjoint{Float64, Vector{Float64}}, Λ2::Matrix{ComplexF64}, Xμ::Adjoint{Float64, Vector{Float64}}, X1::Adjoint{Float64, Vector{Float64}}, X2::Matrix{Float64}, μ::Float64) 
        ζ'*Λ2*ζ + ζ[1]*Λ1*ζ + 1im*ζ'*X2*η + 1im*η[1]*X1*ζ + 1im*μ*Xμ*ζ
    end

    nacc = 0
    nrej = 0
    for e=1:ne
        for i in shuffle(nζs), j in shuffle(nηs)  # goes in shuffled "order"
        # for i in (nζs), j in (nηs)  # goes in order
            ζ0, η0 = ζ[i], η[j]  # save old config
            ΦSB0 = ΦSB(ζ, η, Λ1, Λ2, Xμ, X1, X2, μ)  # compute old action
            ζ[i], η[j] = rand(vζ), rand(vη)  # attempt update
            ΦSB1 = ΦSB(ζ, η, Λ1, Λ2, Xμ, X1, X2, μ)  # compute updated action
            ΔΦSB = ΦSB1 - ΦSB0  # action difference
            if real(exp(-ΔΦSB)) > 1e-6
                @show exp(-ΔΦSB)
                if rand() < real(exp(-ΔΦSB))  # MC
                    nacc += 1
                else
                    nrej += 1
                    ζ[i], η[j] = ζ0, η0  # restore old config
                end
            end
        end
        accr = nacc/(nacc+nrej)
        @show e, accr
    end

    fig, ax = plt.subplots()
    ax.scatter(nηs, η, label="η")
    ax.scatter(nζs, ζ, label="ζ")
    ax.set_xlabel("j")
    ax.legend()
    fig.savefig("cfg.png")
    plt.close()
# end