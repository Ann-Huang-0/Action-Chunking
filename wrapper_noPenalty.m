nS = 5;
chunk = [3 2 1];
chunk_freq = 0.1;

agent.lrate_V = 0.2;
agent.lrate_p = 0.01;
agent.lrate_theta = 0.2;
beta = [0.5 1 1.5 2 2.5]; % capacity constraint
agent.test = 1;

for b = 1:length(beta)
    agent.beta = beta(b);
    %simdata(b) = sim_achunk_generalized(nS, chunk, chunk_freq, agent);
    simdata(b) = sim_noPenalty(nS, chunk, chunk_freq, agent);
    test(b) = simdata(b).test;
end

beta = [0.5 1.0 1.5 2.0 2.5];
bmap = plmColors(length(beta), 'set2');
figure; 
bar([[simdata.chooseC1];[simdata.chooseA3]]');
xticks([1:5])
set(gca, 'XTickLabel', num2cell(beta))
xlabel('\beta')
ylabel('p(choose A|S_3)')
legend('C_1','A_3','Location','Northwest');
legend('boxoff')
box off
