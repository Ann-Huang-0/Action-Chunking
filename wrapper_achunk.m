%% Wrapper function to simulate the experiment for diff state spaces
%% 
rng(2);
withinChunkUpdate = 0;

stateSpace = [3 6 9];
chunk = [3 2 1];
beta = [0.5 1 1.5 2 2.5];
numBetas = length(beta);
chunk_freq = 0.1;

agent.lrate_V = 0.2;
agent.lrate_p = 0.01;
agent.lrate_theta = 0.2;
agent.test = 1;
beta = [0.5 1 1.5 2 2.5]; % capacity constraint

for i = 1:length(stateSpace)
    for b = 1:length(beta)
        agent.beta = beta(b);
        experiment(i,b) = sim_achunk_generalized(stateSpace(i), chunk, chunk_freq, agent, withinChunkUpdate);
    end
end 

pAS = zeros(length(stateSpace), numBetas);
pCS = zeros(length(stateSpace), numBetas);
pS3 = zeros(length(stateSpace), numBetas);
rt_train = zeros(length(stateSpace), numBetas);
rt_test = zeros(length(stateSpace), numBetas);

for i = 1:length(stateSpace)
    for j = 1:numBetas
        simdata = experiment(i,j);
        pAS(i,j) = simdata.chooseA3;
        pCS(i,j) = simdata.chooseC1;
        %pAS(i,j) = simdata.pA;
        %pCS(i,j) = simdata.pC;
        pS3(i,j) = sum(simdata.state==3)/1800;
        rt_train(i,j) = simdata.rt;
        rt_test(i,j)  = simdata.test.rt;
    end
end

%% Plot p(choose A|S), p(choose C|S)
%% 
bmap = plmColors(length(stateSpace), 'set2');
hold on;
bar(beta, pAS');
xlabel("\beta");
ylabel('p(choose A_3|S_3)');
legend('N=3','N=6','N=9','Location','north');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/ac_selectA', '.jpeg']);

bmap = plmColors(length(stateSpace), 'set2');
figure; hold on;
bar(beta, pCS');
xlabel("\beta");
ylabel('p(choose C|S_3)');
legend('N=3','N=6','N=9','Location','northeast');    
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/ac_selectC', '.jpeg']);

%% Stack plot
%%

pCS_stack = pCS ./ (pCS+pAS) / 5;
pAS_stack = pAS ./ (pCS+pAS) / 5;


bmap = plmColors(numBetas, 'b');
bar(categorical({'3' '6' '9'}), pCS_stack', 'stacked');
%set(gca,'XTickLabel',['3';'6';'9']);
xlabel("Number of primitive states");
ylabel('Normalized p(choose C|S_3)');
legend('\beta=0.5', '\beta=1', '\beta=1.5', '\beta=2', '\beta=2.5', 'Location','north')
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/ac_selectC_stack', '.jpeg']);

figure; hold on;
bmap = plmColors(numBetas, 'b');
bar(categorical({'3' '6' '9'}), pAS_stack', 'stacked');
%set(gca,'XTickLabel',['3';'6';'9']);
xlabel("Number of primitive states");
ylabel('Normalized p(choose A_3|S_3)');
legend('\beta=0.5', '\beta=1', '\beta=1.5', '\beta=2', '\beta=2.5', 'Location','northeast')
legend('boxoff')
box off
exportgraphics(gcf,[pwd, '/figures/ac_selectA_stack', '.jpeg']);


%% Plot action slips
%% 
actionSlips = zeros(length(stateSpace), numBetas);
for i = 1:length(stateSpace)
    nS = stateSpace(i);
    for j = 1:numBetas
        simdata = experiment(i,j);
        actionSlips(i,j) = sum(simdata.test.state == chunk(1) & simdata.test.action==1+nS)/sum(simdata.test.state == chunk(1));
    end
end

bmap = plmColors(length(stateSpace), 'set2');
bar(beta, actionSlips');
xlabel("\beta");
ylabel('% Action slips (Test)');
legend('N=3','N=6','N=9','Location','northeast');    
legend('boxoff')
box off

exportgraphics(gcf,[pwd, '/figures/ac_ActionSlips', '.jpeg']);


%% Average reaction time
%% 
figure; hold on;
bar(beta, rt_train);

bar(beta, rt_test);


%% Reward Complexity Tradeoff
%%   
figure; hold on;
bmaps = ['b' 'r' 'g'];

for i = 1:length(stateSpace)
    bmap = plmColors(length(beta),bmaps(i));
    policy = zeros(1,length(beta));
    reward = zeros(1,length(beta));
    for j = 1:length(beta)
        policy(j) = mean([experiment(i,j).KL]);
        reward(j) = mean([experiment(i,j).reward]);
        plot(mean([experiment(i,j).KL]), mean([experiment(i,j).reward]),'.','Color',bmap(j,:),'MarkerSize',50);
    end
    polynomial = polyfit(policy, reward, 2);
    fitted = polyval(polynomial, policy);
    plot(policy, fitted, 'Color',plmColors(1,bmaps(i)));
    %legend({'','','','','',['N=', int2str(stateSpace(i))]}, 'Location', 'Southeast');
    ylabel('Average reward')
    xlabel('Policy complexity')
    hold on;
     %plot(simdata(i).cost,simdata(i).reward,'o','Color',bmap(i,:),'MarkerSize',10); 
end

legend({'','','','','','N=3','','','','','','N=6','','','','','','N=9'}, 'Location', 'Southeast');

legend('boxoff')
title(l,'\beta')
%prettyplot(20)

set(gcf, 'Position',  [500, 100, 1000, 500])

exportgraphics(gcf,[pwd, '/figures/ac_RCtradeoff', '.jpeg']);
