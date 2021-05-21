function simdata = sim_achunk_generalized(nS, chunk, chunk_freq, agent, withinChunkUpdate)
%{
withinChunkUpdate: (boolean) indicates whether we want to update theta
only after first step of the action chunk (if equals 0), or after each step
of the chunk (if equals 1)
%}

rng(222)

nA = nS + 1;
theta = zeros(nS,nA);                 % policy parameters (13 state-features, 4 actions)
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
clen = 0;

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
        a = nS+1;
        % reward
        r = R(s,chunk(clen));
        rt = rt+ 1/length(chunk); % reaction time
        s = chunk(1);
    end
    
    if a == nS+1 && inChunk == 0 % if you just start a chunk
        inChunk = 1; % turn the chunk on and off
        clen = 1;
        rt = rt+1/length(chunk);
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
    
    if (nargin==4) || (withinChunkUpdate==1) || (withinChunkUpdate==0 && clen <= 1)
    % learning updates
        rpe = agent.beta*r - cost - V(s);                      % reward prediction error
        g = agent.beta*(1 - policy(a));                        % policy gradient
        theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;   % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;
    end  
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.action(t) = a;
        simdata.reward(t) = r;
        simdata.state(t) = s;
        simdata.cost(t) = cost; 
        
        if mod(t,300)==0
            simdata.pC_learn(int16(t/300)) = ...  % moving average
                sum(simdata.state == chunk(1) & simdata.action==1+nS)/sum(simdata.state == chunk(1));
            simdata.pA_learn(int16(t/300)) = ...
                sum(simdata.state == chunk(1) & simdata.action==chunk(1))/sum(simdata.state == chunk(1));
        end
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

simdata.chooseC1 = sum(simdata.state == chunk(1) & simdata.action==1+nS)/sum(simdata.state == chunk(1));
simdata.chooseA3 = sum(simdata.state == chunk(1) & simdata.action==chunk(1))/sum(simdata.state == chunk(1));
simdata.pC = (sum(simdata.state == chunk(1) & simdata.action==1+nS)/sum(simdata.state == chunk(1)))/length(chunk);
simdata.pA = sum(simdata.state == chunk(1) & simdata.action==chunk(1))/sum(simdata.state == chunk(1));

simdata.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2);

clen = 0;
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
            a = 1+nS;
            % reward
            r = R(s,chunk(clen));
            rt = rt+1/length(chunk); % reaction time
            s = chunk(1);
        end
        
        if a == 1+nS && inChunk == 0 % if you just start a chunk
            inChunk = 1; % turn the chunk on and off
            clen = 1;
            rt = rt+1/length(chunk);
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
        
        if (nargin==4) ||...
                (nargin==5 && withinChunkUpdate==1) ||...
                (nargin==5 && withinChunkUpdate==0 && clen <= 1)
        % learning updates
            rpe = agent.beta*r - cost - V(s);                      % reward prediction error
            g = agent.beta*(1 - policy(a));                        % policy gradient
            theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;   % policy parameter update
            V(s) = V(s) + agent.lrate_V*rpe;
        end
        
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.test.action(t) = a;
        simdata.test.reward(t) = r;
        simdata.test.state(t) = s;
        simdata.test.cost(t) = cost;
        
    end % state
    
    simdata.test.slips = sum(simdata.test.state == chunk(1) & simdata.test.action==1+nS)/sum(simdata.test.state == chunk(1));
    simdata.test.chooseA3 = sum(simdata.test.state == chunk(1) & simdata.test.action==chunk(1))/sum(simdata.test.state == chunk(1));
    
    simdata.test.rt = rt/length(state);
end % agent test



    