using Flux: onehotbatch
using BSON
using GZip
using Tar
using ProgressMeter
using DelimitedFiles

const TRAINING_OUT_DIR = "train"
const TEXT_DIR = "train_txt"
const PHONES = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
phn2num = Dict(phone=>i for (i, phone) in enumerate(PHONES))
phn2num["sil"] = 1

"""
    createData(data_dir, out_dir)

Extracts data from files in `data_dir` and saves results in `out_dir`.
"""
function createData(out_dir)

  println("Downloading data...")
  download("https://era.library.ualberta.ca/items/d0dd87c9-8091-467d-8c30-bd23a2a1fb55/download/192e48bf-ffbd-4cb9-9e42-3cde0312994d", "mald.tar.gz")

  print("Decompressing data... ")

  local s
  GZip.open("mald.tar.gz") do fh
    s = read(fh)
  end

  open("mald.tar", "w") do w
    write(w, s)
  end
  rm("mald.tar.gz")
    
  println("DONE")
  print("Extracting data... ")
  Tar.extract("mald.tar", "extracted_mald")
  rm("mald.tar")
  println("DONE")

  data_dir = "extracted_mald/sR_bson_subset"

  @showprogress for fname in readdir(data_dir)

    BSON.@load joinpath(data_dir, fname) mfccs labs
    x = mfccs
    labels = [phn2num[x] for x in vec(labs)]
    class_nums = collect(1:62)

    y = Int.(onehotbatch(labels, class_nums))
    BSON.@save joinpath(out_dir, fname) x y
  end
  rm("extracted_mald", recursive=true)
end

"""
    createTextData(from_dir, out_dir)

Reads in BSON data from `from_dir` and writes it out as a tab-delimited file in `out_dir`.
"""
function createTextData(from_dir, out_dir)

  @showprogress for fname in readdir(from_dir)

    fname = joinpath(from_dir, fname)
    BSON.@load fname x y
        
    fname = basename(fname)
    fname = joinpath(out_dir, fname)
    writedlm(replace(fname, ".bson" => ".txt"), x, '\t')
    writedlm(replace(fname, ".bson" => "_labs.txt"), y, '\t')
  end
end

if ! isdir(TRAINING_OUT_DIR) mkdir(TRAINING_OUT_DIR) end
createData(TRAINING_OUT_DIR)

println("Converting to text")
if ! isdir(TEXT_DIR) mkdir(TEXT_DIR) end
createTextData(TRAINING_OUT_DIR, TEXT_DIR)