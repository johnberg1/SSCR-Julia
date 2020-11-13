using HDF5
function simpleRead(filePath,exIdx)
    fid = h5open(filePath,"r")
    example = fid["$(exIdx)"]
    images = read(example["images"])
    utterences = read(example["utterences"])
    objects = read(example["objects"])
    coords = read(example["coords"])
    scene_id = read(example["scene_id"])
    utt_tokenized = [split(t) for t in utterences]
    lengths = [length(t) for t in utt_tokenized]
    (coords,images,objects,scene_id,utterences,utt_tokenized,lengths)
end
coords,images,objects,scene_id,utterences,utt_tokenized,lengths = simpleRead("codraw_val.h5",150);

f = open("glove_codraw_iclevr.txt");
lines = readlines(f)
glove = Dict()
for line in lines
    splitline = split(line)
    word = splitline[1]
    embedding = [parse(Float32,val) for val in splitline[2:end]]
    glove[word] = embedding
end

turns_word_embeddings = zeros((length(utterences), maximum(lengths), 300))

for (i, turn) in enumerate(utt_tokenized)
    for (j, w) in enumerate(turn)
        turns_word_embeddings[i, j,:] = glove[w]
    end
end
