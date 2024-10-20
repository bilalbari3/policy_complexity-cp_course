all_reward = [];
for repetition = 1:100
    fprintf('On repetition %i\n', repetition)

% task parameters
P_reward_short = 0.2; % reward probability for correct identification of short stimulus
P_reward_long = 0.6; % reward probability for correct identification of long stimulus
n_trials = 300;
P_stimulus = 0.5;

% initialize variables
Q_mat = NaN(2,2,n_trials); % [ states x actions x trials ]
trial_type = binornd(1, P_stimulus, 1, n_trials) + 1; % 1 for short trial, 2 for long trial
choice = NaN(1, n_trials); % 0 for short, 1 for long
reward = NaN(1, n_trials);

% initialize parameters
alpha_learn = 0.3;
beta = 1;
Q_mat(:,:,1) = zeros(2,2); % initialize first trial Q values as 0

for t = 1:n_trials
    Q_mat(:, :, t + 1) = Q_mat(:, :, t); % carry forward most Q values
    if trial_type(t) == 1 % short stimulus
        choice_probability_long = exp(beta*Q_mat(1, 2, t)) ./ ...
            (exp(beta*Q_mat(1, 1, t)) + exp(beta*Q_mat(1, 2, t)));
        choice_long = binornd(1, choice_probability_long);
        if choice_long == 0 % correct choice; short stimulus, picked short
            reward(t) = binornd(1, P_reward_short);
            RPE = reward(t) - Q_mat(1, 1, t);
            Q_mat(1, 1, t+1) = Q_mat(1, 1, t) + alpha_learn*RPE;
        elseif choice_long == 1 % incorrect choice
            reward(t) = 0;
            RPE = reward(t) - Q_mat(1, 2, t); % always 0
            Q_mat(1, 2, t+1) = Q_mat(1, 2, t) + alpha_learn*RPE; % update is always 0
        end
    elseif trial_type(t) == 2 % long stimulus
        choice_probability_long = exp(beta*Q_mat(2, 2, t)) ./ ...
            (exp(beta*Q_mat(2, 1, t)) + exp(beta*Q_mat(2, 2, t)));
        choice_long = binornd(1, choice_probability_long);
        if choice_long == 0 % incorrect choice
            reward(t) = 0;
            RPE = reward(t) - Q_mat(2, 1, t); % always 0
            Q_mat(2, 1, t+1) = Q_mat(2, 1, t) + alpha_learn*RPE; % update is always 0
        elseif choice_long == 1 % correct choice; long stimulus, picked long
            reward(t) = binornd(1, P_reward_long);
            RPE = reward(t) - Q_mat(2, 2, t);
            Q_mat(2, 2, t+1) = Q_mat(2, 2, t) + alpha_learn*RPE;
        end
    end
end
all_reward = [all_reward; reward];

end


%% plot it
figure; hold on
plot(squeeze(Q_mat(1, 1, :)))
plot(squeeze(Q_mat(2, 2, :)))



%% add perseveration component
all_reward = [];
all_policy_complexity = [];
for repetition = 1:100
    fprintf('On repetition %i\n', repetition)

% task parameters
P_reward_short = 0.2; % reward probability for correct identification of short stimulus
P_reward_long = 0.6; % reward probability for correct identification of long stimulus
n_trials = 300;
P_stimulus = 0.5;

% initialize variables
Q_mat = NaN(2,2,n_trials); % [ states x actions x trials ]
trial_type = binornd(1, P_stimulus, 1, n_trials) + 1; % 1 for short trial, 2 for long trial
choice = NaN(1, n_trials); % 0 for short, 1 for long
reward = NaN(1, n_trials);
Pa = NaN(2,n_trials); % perseveration

% initialize parameters
alpha_learn = 0.3;
alpha_persev = 0.09;
beta = 10;
Q_mat(:,:,1) = zeros(2,2); % initialize first trial Q values as 0
Pa(:,1) = ones(2,1)*0.5;

for t = 1:n_trials
    Q_mat(:, :, t + 1) = Q_mat(:, :, t); % carry forward most Q values
    if trial_type(t) == 1 % short stimulus
        choice_probability_long = exp(beta*Q_mat(1, 2, t) + log(Pa(2,t))) ./ ...
            (exp(beta*Q_mat(1, 1, t) + log(Pa(1,t))) + exp(beta*Q_mat(1, 2, t) + log(Pa(2,t))));
        choice_long = binornd(1, choice_probability_long);
        if choice_long == 0 % correct choice; short stimulus, picked short
            reward(t) = binornd(1, P_reward_short);
            RPE = reward(t) - Q_mat(1, 1, t);
            Q_mat(1, 1, t+1) = Q_mat(1, 1, t) + alpha_learn*RPE;
        elseif choice_long == 1 % incorrect choice
            reward(t) = 0;
            RPE = reward(t) - Q_mat(1, 2, t); % always 0
            Q_mat(1, 2, t+1) = Q_mat(1, 2, t) + alpha_learn*RPE; % update is always 0
        end
    elseif trial_type(t) == 2 % long stimulus
        choice_probability_long = exp(beta*Q_mat(2, 2, t) + log(Pa(2,t))) ./ ...
            (exp(beta*Q_mat(2, 1, t) + log(Pa(1,t))) + exp(beta*Q_mat(2, 2, t) + log(Pa(2,t))));
        choice_long = binornd(1, choice_probability_long);
        if choice_long == 0 % incorrect choice
            reward(t) = 0;
            RPE = reward(t) - Q_mat(2, 1, t); % always 0
            Q_mat(2, 1, t+1) = Q_mat(2, 1, t) + alpha_learn*RPE; % update is always 0
        elseif choice_long == 1 % correct choice; long stimulus, picked long
            reward(t) = binornd(1, P_reward_long);
            RPE = reward(t) - Q_mat(2, 2, t);
            Q_mat(2, 2, t+1) = Q_mat(2, 2, t) + alpha_learn*RPE;
        end
    end
    
    % update choice probabilities
    Pa(2,t+1) = Pa(2,t) + alpha_persev*(choice_probability_long - Pa(2,t));
    Pa(1,t+1) = 1 - Pa(2,t+1);

    % update choice vector
    choice(t) = choice_long;
end
all_reward = [all_reward; reward];
all_policy_complexity = [all_policy_complexity; MutualInformation(trial_type', choice')];

end
