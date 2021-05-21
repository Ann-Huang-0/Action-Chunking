actionId = linspace(1, nA, nA);
bar([1 2 3], pAS(:,15));
xlabel('state');

figure; hold on;
state1 = [1 4 5 10 11];
state1_cat = categorical({'1', '12', '13', '123', '132'});
pAS = simdata.pas;
pAS1 = pAS(:, state1);
bar(state1_cat, pAS1');
xlabel('Action');
legend('State 1', 'State 2', 'State 3', 'Location', 'north');

figure; hold on;
state2 = [2 6 7 12 13];
state2_cat = categorical({'2', '21', '23', '213', '231'});
pAS2 = pAS(:, state2);
bar(state2_cat, pAS2');
xlabel('Action');
legend('State 1', 'State 2', 'State 3', 'Location', 'north');

figure; hold on;
state3 = [3 8 9 14 15];
state3_cat = categorical({'3', '31', '32', '312', '321'});
pAS3 = pAS(:, state3);
bar(state3_cat, pAS3');
xlabel('Action');
legend('State 1', 'State 2', 'State 3', 'Location', 'north');