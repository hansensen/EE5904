%% Q Learning function
function [policy, Q, reward_tot, state, state_list] = Q_learning(reward, gamma)
Q = zeros(100,4);
Q0 = zeros(100,4);
trial = 1;
% num_reach = 0;
thres = 0.005;
max_ite = 3000;

while trial <= max_ite
    s = 1;
    k = 1;
    alphak = 1;

    while alphak > thres
        n = [1, 2, 3, 4];

        %epsilon_k = 1/k;
        epsilon_k = 100/(100+k);
        %epsilon_k = (1+log(k))/k;
        %epsilon_k = (1+5*log(k))/k;

        alphak = epsilon_k;
        k = k + 1;
        % delete the action -1
        n(reward(s,:) == -1)=[];
        [~,a_max] = max(Q(s,n));
        a_max = n(a_max);
        rand_num = rand(1);
        % exploration
        if rand_num < epsilon_k
            n(n==a_max) =[];
            rand_index = randperm(length(n));
            a = n(rand_index(1)); 
        % exploitation
        else
            a = a_max;
        end

        % Update state        
        s_new = update_state(s, a);

        % Update the Q value
        Q(s,a) = Q(s,a) + alphak * (reward(s,a) + gamma * max(Q(s_new,:)) - Q(s,a));
        s = s_new;

        % counting the number of trials that the state reach 100
%         if s == 100
%             num_reach = num_reach + 1; 
%             break;
%         end
    end

    if trial > 1
    diff(trial) = mean((Q0(:) - Q(:)) .^ 2);
    diff(trial) = 0.05*diff(trial) + 0.95*diff(trial-1);
    % Check whether the Q converge    
    if diff(trial) < thres
    break
    end
    end

    Q0 = Q;
    trial = trial + 1;
end

[~,policy] = max(Q,[],2);

[state, reward_tot, state_list] = plot_graph(policy,reward);
end

function s_new = update_state(s, a)
    % Update state
    switch a
    case 1
    s_new = s-1;
    case 2
    s_new = s+10;
    case 3
    s_new = s+1;
    case 4
    s_new = s-10;
    end
end

function [state, reward_tot, state_list] = plot_graph(policy,reward)
    %% Plot the path
    figure;
    grid on;
    axis([0 10 0 10]);

    state = 1;
    reward_tot = 0;
    n = 0;

    while((state < 100) && (n < 100))
    [y,x] = ind2sub([10,10],state);
    if policy(state) == 1
    text(x - 0.5, 10 - y + 0.5, '^','interpreter','none')
    state = state - 1;
    elseif policy(state) == 2
    text(x - 0.5, 10 - y + 0.5, '>','interpreter','none')
    state = state + 10;
    elseif policy(state) == 3
    text(x - 0.5, 10 - y + 0.5, 'v','interpreter','none')
    state = state + 1;
    else
    text(x - 0.5, 10 - y + 0.5, '<','interpreter','none')
    state = state - 10;
    end
    reward_tot = reward_tot + reward(state, policy(state));
    n = n + 1;
    state_list(n) = state;
    end
end
