addpath('helper')

%%
Q = [0.2   0; % [ states x actions ]
       0 0.6];
Ps = ones(1,2)/2;
beta = logspace(-1,2,100); % beta values

[information, reward, Pa, policy] = blahut_arimoto(Ps, Q, beta);

plot(information, reward, 'linewidth', 4)
xlabel('Policy complexity (bits)')
ylabel('Reward (%)')
set(gca,'fontsize',18)

policy{1} % lowest complexity policy
policy{end} % highest complexity policy