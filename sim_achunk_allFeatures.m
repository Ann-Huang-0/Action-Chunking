function simdata = sim_achunk_allFeatures(agent)
%% Set parameters

rng(0);

nS = 3;
nA = 7;
theta = zeros(nS, nA);
V = zeros(nS, 1);
p = ones(1, nA) / nA;

blockstruct.chunk  = [3 2 1];
blockstruct.length = 1800;
blockstruct.chunk_freq = 0.1;
state = sim_block(nS, blockstruct);   % generate state sequence

actionSet = {1, 2, 3, [1 2 3], [1 3 2], [1 2], [1 3];
             1, 2, 3, [2 1 3], [2 3 1], [2 1], [2 3];
             1, 2, 3, [3 1 2], [3 2 1], [3 1], [3 2]};

R = [1 0 0 1 1 1 1;
     0 1 0 1 1 1 1;
     0 0 1 1 1 1 1];                  % hard-code the reward

 %% Train
 %%
 rt = 0;
 inChunk = 0;            % boolean variable indicates whether in a chunk  
 
 for t = 1:length(state)
     s = state(t);       % sample the current state
     
     if inChunk == 0      % if not in a chunk, select an action
         d = agent.beta*theta(s,:) + log(p);
         logpolicy = d - logsumexp(d);
         policy = exp(logpolicy);       % softmax
         a = fastrandsample(policy);    % sample action
         
         r = R(s,a);                    % get next reward
         rt = rt + 1;
         
     else                 % still in a chunk
         a = simdata.action(t-1);
         option = cell2mat(actionSet(s,a));                          % find out the current action in a chunk
         r = R(s, option(clen));
         rt = rt + 1/length(option);
         s = option(1);
     end
     
     if nS < a && inChunk==0  % if just started a chunk
         inChunk = 1;
         clen = 1;
         option = cell2mat(actionSet(s,a));
         rt = rt + 1/length(option);
     end
     
     cost = logpolicy(a) - log(p(a));    % policy complexity cost
     
     if inChunk == 1
         if clen > 1
             cost = 0;
         end
         clen = clen + 1;
         option = cell2mat(actionSet(a));
         if clen > length(option)
             inChunk = 0;
             clen = 0;
         end
     end
     
     % learning updates
     rpe = agent.beta*r - cost - V(s);                      % reward prediction error
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
    policy = exp(logpolicy);    
    simdata.pas(i,:) = policy;
end

simdata.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2);

%% Test
%%
if agent.test == 1
    state = state(randperm(length(state))); % shuffle states
    
    rt = 0;
    inChunk = 0;            % boolean variable indicates whether in a chunk
    
    for t = 1:length(state)
        s = state(t);       % sample the current state
        
        if inChunk == 0      % if not in a chunk, select an action
            d = agent.beta*theta(s,:) + log(p);
            logpolicy = d - logsumexp(d);
            policy = exp(logpolicy);       % softmax
            a = fastrandsample(policy);    % sample action
            
            r = R(s,a);                    % get next reward
            rt = rt + 1;
            
        else                 % still in a chunk
            a = simdata.test.action(t-1);
            option = cell2mat(actionSet(a));                          % find out the current action in a chunk
            r = R(s, option(clen));
            rt = rt + 1/length(option);
            s = option(1);
            
        end
        
        if (nS < a) && inChunk==0  % if just started a chunk
            inChunk = 1;
            clen = 1;
            option = cell2mat(actionSet(a));
            rt = rt + 1/length(option);
        end
        
        cost = logpolicy(a) - log(p(a));    % policy complexity cost
        
        if inChunk == 1
            if clen > 1
                cost = 0;
            end
            clen = clen + 1;
            option = cell2mat(actionSet(a));
            if clen > length(option)
                inChunk = 0;
                clen = 0;
            end
        end
        
        % learning updates
        rpe = agent.beta*r - cost - V(s);                      % reward prediction error
        g = agent.beta*(1 - policy(a));                        % policy gradient
        theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;   % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;
        
        p = p + agent.lrate_p*(policy - p); 
        p = p./nansum(p);        % marginal update
        simdata.test.action(t) = a;
        simdata.test.reward(t) = r;
        simdata.test.state(t) = s;
        simdata.test.cost(t) = cost;
    end
    
    simdata.test.rt = rt/length(state);
end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    