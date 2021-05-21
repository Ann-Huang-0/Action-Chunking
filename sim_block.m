function block = sim_block(N_states, blockstruct)

chunk     = blockstruct.chunk;
block_len = blockstruct.length;
chunkFreq = floor(blockstruct.chunk_freq * block_len);  % the num of times the chunk occurs

% calculate the out-of-chunk occurences of each state 
occurence = ones(1, N_states) * floor(block_len / N_states);
for i = chunk
    occurence(i) = occurence(i) - chunkFreq;
end

% randomly distribute out-of-chunk states
block = ones(1, sum(occurence));
act_hist = ones(1, sum(occurence));
shuffle = randperm(sum(occurence));
idx     = [0 cumsum(occurence)];

for state = 1:N_states
    shuffled_idx = shuffle(idx(state)+1 : idx(state+1));
    block(shuffled_idx) = state;
    act_hist(shuffled_idx) = state;
end
    
% randomly insert chunks into out-of-chunk states
act_hist = string(act_hist);
insertion = randperm(sum(occurence), chunkFreq);
for i = 1:length(insertion)
    breakpoint = insertion(i) + length(chunk)*(i-1);
    block = [block(1:breakpoint) chunk block(breakpoint+1:end)];
    act_hist = [act_hist(1:insertion(i)) "c" act_hist(insertion(i)+1:end)];
end
    
% quality control
state_dist = zeros(1, N_states);
for state = 1:N_states
    state_dist(state) = sum(block==state);
end
disp(state_dist);

    