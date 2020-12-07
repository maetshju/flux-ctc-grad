# GPU impelmentation

# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using Flux
using Statistics
using CUDA

const MAX_THREADS = 256

function log_plus_f(p1, p2)
  
  isinf(p1) && return p2
  isinf(p2) && return p1

  if p1 < p2
    p1, p2 = p2, p1
  end

  return p1 + CUDA.log(1+CUDA.exp(p2 - p1))
end

function countRepeats(A)
  repeats = 0
  for (i,elem) in enumerate(A)
    if i > 1 && A[i] == A[i-1]
      repeats += 1
    end
  end
  return repeats
end

function computeAlphaKernel(probs, labelSize, uttLength, repeats, labelsWithoutBlanks, labelsWithBlanks, alpha, blankLabel)
  
  tid = threadIdx().x
  L = labelSize
  T = uttLength
  S = length(labelsWithBlanks)
  
  if L + repeats > T
    return nothing
  end
  
  labels = labelsWithBlanks

  # Corner-case checking
  start = (L + repeats <= T) ? 0 : 1
  last = S > 1 ? 2 : 1
  
  # Fill in first column (time step)
  i = tid
  while i <= last - start
    alpha[start+i, 1] = probs[labels[start+i], 1]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in coefficients for each time step
  for t=2:T
    
    # Corner-case checking
    if tid == 1 && !(1 < S - 2*(T-t) - 1)
      if start == 0
        alpha[1, t] = probs[blankLabel, t] + alpha[1, t-1]
      elseif start == 1
        alpha[1, t] = alpha[1, t-1]
      end
    end
    
    sync_threads()
    
    # Fill in coefficients for each label class in the target output sequence;
    # each thread will process the calculations for one class
    idx = tid+1
    while idx <= S
      
      prevSum = log_plus_f(alpha[idx, t-1], alpha[idx-1, t-1])
      
      if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx-2]
        prevSum = log_plus_f(prevSum, alpha[idx-2, t-1])
      end
      
      if idx < S - 2*(T-t) - 1
        alpha[idx, t] = -Inf32
      else
        alpha[idx, t] = prevSum + probs[labels[idx], t]
      end
    
      idx += blockDim().x
    end
    
    sync_threads()
  end
  return nothing
end

function computeBetasAndGradKernel(probs, labelSize, uttLength,
                  repeatsInLabel, labelsWithBlanks,
                  alphas, beta, output, accum,
                  grad, blankLabel)
  
  tid = threadIdx().x
  L = labelSize
  T = uttLength
  S = 2*L + 1
  repeats = repeatsInLabel
  
  labels = labelsWithBlanks
  
  if (L+repeats) > T
    return nothing
  end
  
  # Corner-case checking
  start = S > 1 ? S-2 : 0
  last = L + repeats < T ? S : S-1
  
  sync_threads()
  
  i = tid
  
  # Calculate coefficients for last column (time step)
  # then determine alpha and beta product
  while i <= last - start + 1
    beta[i+start, T] = 0
    output[i+start, T] = beta[i+start, T] + alphas[i+start, T]
    i += blockDim().x
  end
  
  sync_threads()
  
  # Fill in `accum` for last column (time step)
  if tid == 1    
    for i=1:S
      labelIdx = labels[i]
      accum[labelIdx, T] = log_plus_f(accum[labelIdx, T], output[i, T])
    end
  end
  
  sync_threads()
  
  # Fill in `grad` for last column (time step)
  idx = tid
  while idx <= size(grad, 1)
    
    s = -Inf32
    
    for i=1:S
      s = log_plus_f(s, output[i, T])
    end
    
    # ∂L/∂a (where a is activation before logsoftmax)
    grad[idx, T] = CUDA.exp(probs[idx, T]) - CUDA.exp(accum[idx, T] - s)
    idx += blockDim().x
  end
  
  sync_threads()
  
  # Fill in the rest of the coefficients
  t = T-1
  while t >= 1
    if t < T
      
      idx = tid
      # while idx <= S-1
      while idx <= S
        
        nextSum = beta[idx, t+1] + probs[labels[idx], t+1]

        if idx < S

          nextSum = log_plus_f(nextSum,
            beta[idx+1, t+1] + probs[labels[idx+1], t+1])
        end
        
        if labels[idx] != blankLabel && idx != S-1 && labels[idx] != labels[idx+2]
          nextSum = log_plus_f(nextSum,
            beta[idx + 2, t+1] + probs[labels[idx+2], t+1])
        end
        
        if idx > 2*t
          beta[idx, t] = -Inf32
        else
          beta[idx, t] = nextSum
            
        end
        
        idx += blockDim().x
      end
    
      sync_threads()
      
      if tid == 1 && last == S
        beta[S, t] = beta[S, t] + probs[blankLabel, t+1]
      end
      
      sync_threads()
      
      idx = tid
      while idx <= S
        output[idx, t] = alphas[idx, t] + beta[idx, t]
        idx += blockDim().x
      end
      
      sync_threads()
    end
    
    
    sync_threads()
    
    # Calculate accumulated alpha-beta products for each label class for
    # each time step; used in calculating gradients
    if tid == 1      
      for i=1:S
        labelIdx = labels[i]
        accum[labelIdx, t] = log_plus_f(accum[labelIdx, t], output[i, t])
      end
    end
    
    sync_threads()
    
    idx = tid
    
    # Calculate gradients
    while idx <= size(grad, 1)
      
      s = -Inf32
      
      for i=1:S
        s = log_plus_f(s, output[i, t])
      end
      
      # ∂L/∂a (where a is activation before logsoftmax)
      grad[idx, t] = CUDA.exp(probs[idx, t]) - CUDA.exp(accum[idx, t] - s)
      idx += blockDim().x
    end
    
    sync_threads()
    
    t -= 1
    sync_threads()
  end

  return nothing
end

# methods for `ctc_` helper function
ctc(ŷ::CuArray, y::Array) = ctc_(ŷ, y)[1] |> mean
ctc(ŷ::Array, y::CuArray) = ctc_(CuArray(ŷ), collect(y))[1] |> mean
ctc(ŷ::CuArray, y::CuArray) = ctc_(ŷ, collect(y))[1] |> mean
ctc_(ŷ::Array, y::CuArray) =  ctc_(CuArray(ŷ), collect(y))

function ctc_(ŷ::CuArray, y)
  
  ŷ = logsoftmax(ŷ)
  
  blank = size(ŷ, 1)
  labels = [Base.argmax(y[:,i]) for i in 1:size(y, 2)]
  z = F(labels, blank)
  z′ = [blank]
  for label in z
    push!(z′, label)
    push!(z′, blank)
  end
  
  T = size(ŷ, 2)
  U′ = 2*length(z) + 1
  
  alphas = CUDA.fill(log(zero(ŷ[1])), U′, T)
  betas = CUDA.fill(log(zero(ŷ[1])), U′, T)
  output = CUDA.fill(log(zero(ŷ[1])), U′, T)
  
  nRepeats = countRepeats(labels)
  nThreads = min(U′, MAX_THREADS)

  @cuda blocks=1 threads=nThreads computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z), CuArray(z′), alphas, blank)
  
  grads = CUDA.fill(log(zero(ŷ[1])), size(ŷ))
  accum = CUDA.fill(log(zero(ŷ[1])), size(ŷ))
  
  @cuda blocks=1 threads=nThreads computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CuArray(z′), alphas, betas, output, accum, grads, blank)
  
  ls = collect(output)
  ls = vec(-1 .* [logsum(ls[:,i]) for i in 1:size(ls, 2)])

  ŷ = alphas = betas = output = accum = nothing
  return ls, grads
end
