function simdata = sim_noPenalty(nS, chunk, chunk_freq, agent)

rng(24);

nA = nS + 1;
theta = zeros(nS,nA);                 % policy parameters 
V = zeros(nS,1);                      % state value weights
p = ones(1,nA)/nA;                    % marginal action probabilities

blockstruct.chunk  = chunk;
blockstruct.length = 1800;
blockstruct.chunk_freq = chunk_freq;

state = sim_block(nS, blockstruct);   % generate state sequence
R_chunk = zeros(nS, 1);               % create the matrix for reward
R_chunk(chunk(1)) = 1;
R = [eye(nS) R_chunk];

%%

rt = 0;
inChunk = 0; % indicates whether it is in a chunk
for t = 1:length(state)
    s = state(t);  % sample start location
    
    % policy
    if inChunk == 0  % not in a chunk, sample action and get reward
        d = agent.beta*theta(s,:) + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax
        a = fastrandsample(policy);   % sample action
        
        % reward
        r = R(s,a);
        rt = rt+1; % reaction time
        
    else % still in chunk
        a = 5;
        % reward
        r = R(s,chunk(clen));
        rt = rt+0.3; % reaction time
        s = 3;
    end
    
    if a == 5 && inChunk == 0 % if you just start a chunk
        inChunk = 1; % turn the chunk on and off
        clen = 1;
        rt = rt+0.3;
    end
    
    cost = logpolicy(a) - log(p(a));    % policy complexity cost
    
    if inChunk == 1
        if clen > 1 % if action is tere
            cost = 0; % no cost if executing chunk
        end
        clen = clen + 1;
        
        if clen > length(chunk)
            inChunk = 0;  % turn chunk off when action sequence is over
            clen = 0;
        end
        
    end
    
    % learning updates
    rpe = r - V(s);                      % reward prediction error
    g = agent.beta*(1 - policy(a));                        % policy gradient
    theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;   % policy parameter update
    V(s) = V(s) + agent.lrate_V*rpe;
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
    simdata.action(t) = a;
    simdata.reward(t) = r;
    simdata.state(t) = s;
    simdata.cost(t) = cost;
    
end

simdata.rt = rt/length(state);
simdata.V = V;
simdata.theta = theta;
simdata.pa = p;

for i = 1:nS
    d = agent.beta*theta(i,:) + log(p);
    logpolicy = d - logsumexp(d,2);
    policy = exp(logpolicy);    % softmax
    simdata.pas(i,:) = policy;
end

simdata.chooseC1 = sum(simdata.state == 3 & simdata.action==5)/sum(simdata.state == 3);
simdata.chooseA3 = sum(simdata.state == 3 & simdata.action==3)/sum(simdata.state == 3);

simdata.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2);


if agent.test == 1
    
    state = state(randperm(length(state))); % shuffle states
    
    rt = 0;
    inChunk = 0;
    for t = 1:length(state)
        s = state(t);  % sample start location
        
        % policy
        if inChunk == 0
            d = agent.beta*theta(s,:) + log(p);
            logpolicy = d - logsumexp(d);
            policy = exp(logpolicy);    % softmax
            a = fastrandsample(policy);   % sample action
            
            % reward
            r = R(s,a);
            rt = rt+1; % reaction time
            
        else % still in chunk
            a = 5;
            % reward
            r = R(s,chunk(clen));
            rt = rt+0.3; % reaction time
            s = 3;
        end
        
        if a == 5 && inChunk == 0 % if you just start a chunk
            inChunk = 1; % turn the chunk on and off
            clen = 1;
            rt = rt+0.3;
        end
        
        
        cost = logpolicy(a) - log(p(a));    % policy complexity cost
        
        if inChunk == 1
            if clen > 1 % if action is tere
                cost = 0; % no cost if executing chunk
            end
            clen = clen + 1;
            
            if clen > length(chunk)
                inChunk = 0;  % turn chunk off when action sequence is over
                clen = 0;
            end
            
        end
        
        
        % learning updates
        rpe = r - V(s);                                        % reward prediction error
        g = agent.beta*(1 - policy(a));                        % policy gradient
        theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;   % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;
        
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.test.action(t) = a;
        simdata.test.reward(t) = r;
        simdata.test.state(t) = s;
        simdata.test.cost(t) = cost;
        
    end % state
    
    simdata.test.slips = sum(simdata.test.state == 3 & simdata.test.action==5)/sum(simdata.test.state == 3);
    simdata.test.chooseA3 = sum(simdata.test.state == 3 & simdata.test.action==3)/sum(simdata.test.state == 3);

    simdata.test.rt = rt/length(state);
end % agent test


end