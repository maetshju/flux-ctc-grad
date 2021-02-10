using Flux
using BSON
using LinearAlgebra
using Statistics
using Random
using Zygote: @adjoint
using Zygote: @nograd
using Zygote
using ProgressBars
using Distributions
using Flux.Losses: ctc_loss

Random.seed!(1)

const TRAINDIR = "train"

const EPOCHS = 100
const BATCH_SIZE = 1
const VAL_SIZE = 500

losses = []

forward = LSTM(39, 100)
backward = LSTM(39, 100)
output = Dense(200, 62)

const NOISE = Normal(0, 0.6)

function m(x)
  h0f = forward.(x)
  h0b = Flux.flip(backward, x)
  h0 = vcat.(h0f, h0b)
  o = output.(h0)
  return o
end

function loss(x, y)
  x = addNoise(x)
  Flux.reset!((forward, backward))
  yhat = m(x)
  yhat = reduce(hcat, yhat)
  l = ctc_loss(yhat, y)
  addToGlobalLoss(l)
  return l
end

@nograd function addNoise(x)
  return [xI .+ Float32.(rand(NOISE, 39)) for xI in x]
end

function readData(dataDir)
  fnames = readdir(dataDir)
  shuffle!(MersenneTwister(4), fnames)
  Xs = []
  Ys = []

  for fname in fnames
    BSON.@load joinpath(dataDir, fname) x y
    x = [Float32.(x[i,:]) for i in 1:size(x,1)]
    push!(Xs, x)
    push!(Ys, collapse(argmax.(eachcol(y))))
  end

  m = mean(reduce(vcat, Xs))
  st = std(reduce(vcat, Xs))
  for (i, x) in enumerate(Xs)
    Xs[i] = [(xI .- m) ./ st for xI in x]
  end

  return (Xs, Ys)
end

function lev(s, t)
    m = length(s)
    n = length(t)
    d = Array{Int}(zeros(m+1, n+1))

    for i=2:(m+1)
        @inbounds d[i, 1] = i-1
    end

    for j=2:(n+1)
        @inbounds d[1, j] = j-1
    end

    for j=2:(n+1)
        for i=2:(m+1)
            @inbounds if s[i-1] == t[j-1]
                substitutionCost = 0
            else
                substitutionCost = 1
            end
            @inbounds d[i, j] = min(d[i-1, j] + 1, # Deletion
                            d[i, j-1] + 1, # Insertion
                            d[i-1, j-1] + substitutionCost) # Substitution
        end
    end

    @inbounds return d[m+1, n+1]
end

function collapse(seq)
  if isempty(seq) return seq end
  s = [seq[1]]
  for ch in seq[2:end]
    if ch != s[end]
      push!(s, ch)
    end
  end
  filter!(x -> x != 62, s)
  return s
end

# collapses all repetitions into a single item and
# removes blanks; used because data set in its current
# form has one-hot encoded phones for every time step
# instead of just the phones contained in the sequence.
# With a proper re-extraction of the data, this would not
# be necessary.
function alt_collapse(seq)
  if isempty(seq) return seq end
  s = [seq[1]]
  for ch in seq[2:end]
    if ch != s[end]
      push!(s, ch)
    end
  end
  filter!(!=(62), s)
  return s
end

"""
    per(x, y)

Compute the phoneme error rate of the model for input `x` and target `y`. The phoneme error rate
is defined as the Levenshtein distance between the labeling produced by running `x` through
the model and the target labeling in `y`, all divided by the length of the target labeling
in `y`
"""
function per(x, y)
  Flux.reset!((forward, backward))
  yhat = m(x)
  yhat = reduce(hcat, yhat)
  yhat = mapslices(argmax, yhat, dims=1) |> vec |> alt_collapse
  # y = mapslices(argmax, y, dims=1) |> vec |> collapse
  return lev(yhat, y) / length(y)
end

function addToGlobalLoss(x)
	global losses
	push!(losses, x)
end

@adjoint function addToGlobalLoss(x)
	addToGlobalLoss(x)
	return nothing, () -> nothing
end

function main()
  println("Loading files")
  Xs, Ys = readData(TRAINDIR)
  data = collect(zip(Xs, Ys))

  Xs = [d[1] for d in data]
  Ys = [d[2] for d in data]

  println("Beginning training")
  
  data = zip(Xs, Ys) |> collect
  valData = data[1:VAL_SIZE]
  data = data[VAL_SIZE+1:end]

  opt = Momentum(1e-4)

  for i in 1:EPOCHS
    global losses
    losses = []
    println("Beginning epoch $i/$EPOCHS")
    Flux.train!(loss, Flux.params((forward, backward, output)), ProgressBar(data), opt)
    println("Calculating PER...")
    p = mean(map(x -> per(x...), valData))
    println("PER: $(p*100)")

    println("Mean loss: ", mean(losses))
  end
end

main()
