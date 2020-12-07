using PyCall
using Zygote: @adjoint
pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)  
cctc = pyimport("c_ctc")

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

function c_ctc_(yhat, y)
  y = mapslices(argmax, y, dims=1)
  y = collapse(y)
  y = reshape(y, 1, length(y))
	
  l, g = cctc.c_ctc(yhat, Int32.(y .- 1))
  l = l[1]
  return l, g
end

function c_ctc(yhat, y)
  return c_ctc_(yhat, y)[1]
end

@adjoint function c_ctc(yhat, y)
  l, g = c_ctc_(yhat, y)
  return l, Δ -> (Δ .* g, Δ)
end
