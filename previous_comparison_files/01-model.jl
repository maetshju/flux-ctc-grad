using Flux
using BSON
using LinearAlgebra
using Statistics
using Random
using Zygote: @adjoint

Random.seed!(1)

include("ctc.jl")

const TRAINDIR = "train"

const EPOCHS = 300
const BATCH_SIZE = 1

const N_FILES = 400

losses = []

forward = LSTM(39, 100)
backward = LSTM(39, 100)
output = Dense(200, 62)

# m(x) = output.(forward.(x))
function m(x)
  h0f = collect(map(forward, x))
  h0b = reverse(map(backward, reverse(x)))
  h0 = map(i -> vcat(h0f[i], h0b[i]), 1:length(h0f))
  o = collect(map(output, h0))
  return o
end

function loss(x, y)
  Flux.reset!((forward, backward))
  yhat = m(x)
  yhat = reduce(hcat, yhat)
  l = ctc(yhat, y)
  addToGlobalLoss(l)
  return l
end

function readData(dataDir)
  fnames = open(readlines, "shuffled_names.txt")
  fnames = [x * ".bson" for x in fnames]
  fnames = fnames[1:N_FILES]
  Xs = []
  Ys = []

  for fname in fnames
    BSON.@load joinpath(dataDir, fname) x y
    x = [Float32.(x[i,:]) for i in 1:size(x,1)]
    push!(Xs, x)
    push!(Ys, y)
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


"""
    collapse(seq)

Collapses `seq` into a version that has symbols collapsed into one repetition and removes all instances
of the blank symbol.
"""
function collapse(seq)
  s = [x for x in seq if x != 62]
  if isempty(s) return s end
  s = [seq[1]]
  for ch in seq[2:end]
    if ch != s[end] && ch != 62
      push!(s, ch)
    end
  end
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
  yhat = mapslices(argmax, yhat, dims=1) |> vec |> collapse
  y = mapslices(argmax, y, dims=1) |> vec |> collapse
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

  opt = Momentum(1e-4)
  # opt = ADAM(1e-4)

  # yhat = m(data[1]...)
  # Flux.reset!(forward)
  # yhat_m = reduce(hcat, yhat)
  # ctc_g = gradient(ctc, yhat_m, data[1][2])
  # p = params(forward, output)
  # gs = gradient(p) do
  #   loss(data[1]...)
  # end

  # println(ctc_g[:,end])
  # println(output.W[end,:])
  # println(gs[p[6]][end,:])
  # exit()
  
  for i in 1:EPOCHS
    global losses
    losses = []
    println("Beginning epoch $i/$EPOCHS")
    Flux.train!(loss, params((forward, backward, output)), data, opt)
    println("Calculating PER...")
    p = mean(map(x -> per(x...), data))
    println("PER: $(p*100)")

    println("Mean loss: ", mean(losses))
    if p < 0.35 exit() end
  end
end

main()
