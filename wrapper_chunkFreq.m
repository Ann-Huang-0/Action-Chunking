stateSpace = [3 6 9];
chunk = [3 2 1];
chunk_freq = [0.001 0.01 0.05 0.1];
% agent.lrate_V = 0.2;
agent.lrate_V = 0.1;
agent.lrate_p = 0.01;
% agent.lrate_theta = 0.2;
agent.lrate_theta = 0.1;
agent.beta = 1; % capacity constraint
agent.test = 1;
    
withinChunkUpdate = 0;

%% 

pAS = zeros(length(stateSpace), length(chunk_freq));
pCS = zeros(length(stateSpace), length(chunk_freq));

for i = 1:length(stateSpace)
    nS = stateSpace(i);
    for j = 1:length(chunk_freq)
        simdata(i,j) = sim_achunk_generalized(nS, chunk, chunk_freq(j), agent, withinChunkUpdate);
        %pAS(i,j) = simdata(i,j).chooseA3;
        %pCS(i,j) = simdata(i,j).chooseC1;
        %pAS(i,j) = simdata(i,j).pA;
        %pCS(i,j) = simdata(i,j).pC;
        pAS(i,j) = simdata(i,j).pa(3);
        pCS(i,j) = simdata(i,j).pa(4);
    end
end

% plot
bmap = plmColors(length(chunk_freq), 'b');
bar(categorical({'N=3', 'N=6', 'N=9'}), pAS);
xlabel("Number of primitive states");
ylabel('p(choose A_3|S_3), \beta=1');
legend('freq=0.001','freq=0.01','freq=0.05','freq=0.1', 'Location','north');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/freqPAS', '.jpeg']);

figure; hold on;
bmap = plmColors(length(chunk_freq), 'b');
bar(categorical({'N=3', 'N=6', 'N=9'}), pCS);
xlabel("Number of primitive states");
ylabel('p(choose C|S_3), \beta=1');
legend('freq=0.001','freq=0.01','freq=0.05','freq=0.1', 'Location','north');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/freqPCS', '.jpeg']);

%% Plot with fixed state space and differing beta
%%
betas = [0.5 1 1.5 2 2.5];
for i = 1:length(betas)
    agent.beta = betas(i);
    for j = 1:length(chunk_freq)
        simdata(i,j) = sim_achunk_generalized(3, chunk, chunk_freq(j), agent, withinChunkUpdate);
        pAS(i,j) = simdata(i,j).chooseA3;
        pCS(i,j) = simdata(i,j).chooseC1;
    end
end

bmap = plmColors(length(chunk_freq), 'k');
bar(betas, pAS');
xlabel("\beta");
ylabel('p(choose A_3|S_3), N=3');
legend('freq=0.001','freq=0.01','freq=0.05','freq=0.1', 'Location','north');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/freqPAS_diffBeta', '.jpeg']);

figure; hold on;
bmap = plmColors(length(chunk_freq), 'k');
bar(betas, pCS');
xlabel("\beta");
ylabel('p(choose C|S_3), N=3');
legend('freq=0.001','freq=0.01','freq=0.05','freq=0.1', 'Location','north');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/freqPCS_diffBeta', '.jpeg']);

        
        


