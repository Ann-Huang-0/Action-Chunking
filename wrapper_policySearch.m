%% VARYING BETA
%%
withinChunkUpdate = 0;
agent.lrate_V = 0.2;
agent.lrate_p = 0.01;
agent.lrate_theta = 0.2;
agent.test = 1;
beta = [0.2 0.4 0.6 0.8 1];
%beta = [0.5 1.0 1.5 2.0 2.5];

pAS = {};
pA = {};
for i=1:length(beta)
    agent.beta = beta(i);
    simdata(i) = sim_achunk_subunit(3, 0.1, agent, withinChunkUpdate);
    pAS{end+1} = simdata(i).pas;
    pA{end+1} = simdata(i).pa;
end
%% primitive actions VS chunks (marginal)
%%
pA_vs_pC = zeros(2, length(beta));
for i=1:length(beta)
    pA_thisBeta = cell2mat(pA(i));
    pA_vs_pC(1,i) = sum(pA_thisBeta(4:end));
    pA_vs_pC(2,i) = sum(pA_thisBeta(1:3));
end

bar(beta, pA_vs_pC);
legend('p(C)', 'p(A)','location','northwest');
xlabel('\beta');
%% primitive actions VS length-2 chunks VS length-3 chunk (marginal)
%%
pA_vs_PC = zeros(3, length(beta));
for i=1:length(beta)
    pA_thisBeta = cell2mat(pA(i));
    pA_vs_pC(1,i) = sum(pA_thisBeta(6:end));
    pA_vs_pC(2,i) = sum(pA_thisBeta(4:5));
    pA_vs_pC(3,i) = sum(pA_thisBeta(1:3));
end
bar(beta, pA_vs_pC);
legend("A=\{3,2,1\}", 'A=\{\{3,2\},\{2,1\}\}','A=\{\{3\},\{2\},\{1\}\}','location','northwest');
xlabel('\beta');
ylabel('p(choose A)');

%% p(A={3}|S=3) VS p(A={3,2}|S=3) VS p(A={3,2,1}|S=3)
%%
subunits = zeros(3, length(beta));
for i=1:length(beta)
    pAS_thisBeta = cell2mat(pAS(i));
    subunits(1,i) = pAS_thisBeta(3, 3);  % select {3} at state 3
    subunits(2,i) = pAS_thisBeta(3, 5);  % select {3,2} at state 3
    subunits(3,i) = pAS_thisBeta(3, 6);  % select {3,2,1} at state 3
end   

bmap = plmColors(3, 'set2');
X = categorical({'0.2','0.4', '0.6', '0.8', '1.0'});
bar(X, subunits);
legend('A=\{3\}', 'A=\{3,2\}', 'A=\{3,2,1\}','location', 'northwest');
legend('boxoff');
xlabel('\beta');
ylabel('p(choose A|S_3');
box off

bmap = plmColors(length(beta),'b');
X = categorical({'A=\{3\}', 'A=\{3,2\}','A=\{3,2,1\}'});
X = reordercats(X, {'A=\{3\}', 'A=\{3,2\}','A=\{3,2,1\}'});
bar(X,subunits');
legend('\beta=0.2', '\beta=0.4','\beta=0.6','\beta=0.8','\beta=1.0');
xlabel('Action taken at S=3');
ylabel('p(choose A|S_3)');
legend('boxoff');
box off
%% p(A={2}|S=2) VS p(A={2,1}|S=2)
%% 
subunits = zeros(2, length(beta));
for i=1:length(beta)
    pAS_thisBeta = cell2mat(pAS(i));
    subunits(1,i) = pAS_thisBeta(2, 2);  % select {2} at state 2
    subunits(2,i) = pAS_thisBeta(2, 4);  % select {2,1} at state 3
end   

bmap = plmColors(3, 'set2');
X = categorical({'0.2','0.4', '0.6', '0.8', '1.0'});
bar(X, subunits);
legend('A=\{2\}', 'A=\{2,1\}', 'location', 'northwest');
xlabel('\beta');
ylabel('p(choose A|S_2');
legend('boxoff');
box off

bmap = plmColors(length(beta),'b');
X = categorical({'A=\{2\}','A=\{2,1\}'});
X = reordercats(X, {'A=\{2\}','A=\{2,1\}'});
bar(X,subunits');
legend('\beta=0.5', '\beta=1.0','\beta=1.5','\beta=2.0','\beta=2.5');
xlabel('Action taken at S=2');
ylabel('p(choose A|S_2');
legend('boxoff');
box off


%% VARYING STATE SPACE N
%% 
agent.lrate_V = 0.2;
agent.lrate_p = 0.01;
agent.lrate_theta = 0.2;
agent.test = 1;
agent.beta = 0.5;
nS = [3 6 9];

pAS = {};
pA = {};
plot = zeros(3,3);
plot2 = zeros(2,3);

for i = 1:length(nS)
    simdata(i) = sim_achunk_subunit(nS(i), 0.1, agent,withinChunkUpdate);
    pAS{end+1} = simdata(i).pas;
    pA{end+1} = simdata(i).pa;
end

mat1 = cell2mat(pAS(1));mat2 = cell2mat(pAS(2));mat3 = cell2mat(pAS(3));
plot(1,1)=mat1(3,3); plot(2,1)=mat1(3,5); plot(3,1)=mat1(3,6);
plot(1,2)=mat2(3,3); plot(2,2)=mat2(3,8); plot(3,2)=mat2(3,9);
plot(1,3)=mat3(3,3); plot(2,3)=mat3(3,11); plot(3,3)=mat3(3,12);
%normalizedPlot = plot ./ [1/3 1/6 1/9; 1/3 1/6 1/9;1/3 1/6 1/9];

bmap = plmColors(3, 'set2');
bar([3 6 9], plot');
% bar([3 6 9], normalizedPlot');
box off
xlabel('N');
ylabel('p(choose A|S_3');
legend('A=\{3\}', 'A=\{3,2\}', 'A=\{3,2,1\}','location', 'northeast');
legend('boxoff');

plot2(1,1)=mat1(2,2); plot2(2,1)=mat1(2,4);
plot2(1,2)=mat2(2,2); plot2(2,2)=mat2(2,7);
plot2(1,3)=mat3(2,2); plot2(2,3)=mat3(2,10);
bmap = plmColors(2, 'set2');
bar([3 6 9], plot2');
box off
xlabel('N');
ylabel('p(choose A|S_3');
legend('A=\{2\}', 'A=\{2,1\}', 'location', 'northeast');
legend('boxoff');