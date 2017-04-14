function q=ReinforcementLearningGreedy(R, gamma, goalState, alpha, epsilon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original Q Learning by Example code, by Kardi Teknomo 
% (http://people.revoledu.com/kardi/)
%
% Code amended by Ioannis Makris and Andrew Chalikiopoulos
% Code edited by Chadi El Hajj and Muaaz Bin Sarfaraz
% Model for an agent to find shortest path through a 7x7 maze grid
% This algorithm uses a the Epsilon-greedy algorithm to choose the next state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
clc;
format short
format compact

% Four inputs: R, gamma and alpha, epsilon
if nargin<1,
% immediate reward matrix;
     R=csvread('RewardMatrix49.csv');
end
if nargin<2,
    gamma=0.5;              % discount factor
    alpha=0.8;              % learning rate
    epsilon=0.9;             % epsilon value for greedy algorithm
end
if nargin<3
    goalState=25;   
end

q=zeros(size(R));        % initialize Q as zero
q1=ones(size(R))*inf;    % initialize previous Q as big number
count=0;                 % counter
steps=0;                 % counts the number of steps to goal
B=[];                    % matrix to add results of steps and episode count
cumReward=0;             % counter to calculate accumulated reward
exploitCount=0;
exploreCount=0;

for episode=1:50000
    
    state=1;        % Starting state of the agent
    
    
    while state~=goalState            % loop until find goal state
        % select any action from this state using ?-greedy
        x=find(R(state,:)>=0);         % find possible action of this state including punishment
        if size(x,1)>0,
            
            r=rand; % get a uniform random number between 0-1
     
     % choose either explore or exploit
     if r>=epsilon   % exploit
         [~,qmax]=(max(q(state,x(1:end)))); % check for action with highest Q value
         x1 = x(qmax);  % set action with highest Q value as next state
         
         if epsilon>=0.5
            epsilon=epsilon*0.99999; % decrease epsilon
         else
             epsilon=epsilon*0.9999; % decrease epsilon
         end
         
         cumReward=cumReward+q(state,x1); %keep track of cumulative reward for graph
         exploitCount=exploitCount+1;
         %display('The agent exploits.');
         
     else        % explore
             x1=RandomPermutation(x);   % randomize the possible action
             x1=x1(1);                  % select an action (only the first element of random sequence)
             
             if epsilon>=0.5
                epsilon=epsilon*0.99999; % decrease epsilon
             else
                epsilon=epsilon*0.9999; % decrease epsilon
             end
             
             cumReward=cumReward+q(state,x1); %keep track of cumulative reward for graph
             exploreCount=exploreCount+1;
             %display('The agent explores.');
     
     end

        x2 = find(R(x1,:)>=0);   % find possible steps from next step
        qMax=(max(q(x1,x2(1:end)))); % extract qmax from all possible next states
        q(state,x1)= q(state,x1)+alpha*((R(state,x1)+gamma*qMax)-q(state,x1));    % Temporal Difference Error
        state=x1;    % set state to next state
        
        end
           
        if state~=goalState     % keep track of steps taken if goal not reached
            steps=steps+1;
            
        else
            steps=steps+1;
            %episodes=episodes+1; % if goal reach increase episode counter
            A=[episode; steps; cumReward];   % create episodes, steps and cumReward matrix
            B=horzcat(B, A);    % add the new results to combined matrix 
        end
    
    end
        
    % break if convergence: small deviation on q for 1000 consecutive
    if sum(sum(abs(q1-q)))<0.00001 && sum(sum(q >0)) && epsilon<0.01
        if count>1000,
            q1=q;
            %episode  % report last episode
            break % for
        else
            q1=q;
            count=count+1; % set counter if deviation of q is small
        end
    else
        q1=q;
        count=0;  % reset counter when deviation of q from previous q is large
    end
    fprintf('Episode %i completed. The agent required %i steps to reach the goal.The cumulative reward gained is %i.\n', episode, steps, cumReward);
    steps=0;    % reset steps counter to 0
    cumReward=0;    % reset cumReward counter to 0
end

% row 4 in matrix is cumReward/steps taken per episode
%B(4,:) = (B(3,:)./B(2,:));
B(4,:) = rdivide(B(3,:),B(2,:));

%episodes vs cumReward taken averaged against steps taken
%plot(B(1,1 : 5 : end),B(3,1 : 5 : end));

%create a plot of episodes vs steps taken and episodes vs cumReward taken averaged against steps taken
figure % new figure]
grid on
set(gcf,'numbertitle','off','name','Return VS Episode performance');
x1 = B(1,1 : 5 : end)
y1 = B(2,1 : 5 : end)

x2= B(1,1 : 5 : end)
y2 = B(4,1 : 5 : end)

yyaxis left 
plot (x1,y1,'g')
title('Q-Learning Performance')
xlabel('Episodes')
ylabel('Steps')
yyaxis right
plot(x2,y2,'r')
ylabel('Return per steps')
legend('Steps','Return per step')
%[combinedGraph] = plotyy(B(1,1 : 5 : end),B(2,1 : 5 : end),B(1,1 : 5 : end),B(4,1 : 5 : end));




% create a plot of episodes vs cumReward/steps
figure

x1= B(1,1 : 5 : end)
y1 =B(4,1 : 5 : end)
grid on
plot (x1,y1,'r')

%plot(B(1,1 : 5 : end),B(4,1 : 5 : end));
title('Return vs Episodes')
xlabel('Episode')
ylabel('Return')
%grid off
% create a plot of episodes vs steps
figure
grid on
plot(B(1, 1 : 5 : end),B(2, 1 : 5 : end));
title('Steps vs Episodes')
xlabel('Episode')
ylabel('Steps')


%normalize q
g=max(max(q));
if g>0, 
    q=100*q/g;
end

% display the shortest path to the goal
Optimal=[];
state=1;
Optimal=horzcat(Optimal,state);

while state~=goalState
    
         [~,optimal]=(max(q(state,:)));
         state = optimal;
         Optimal=horzcat(Optimal,state);         
end

display('Shortest path:')
display(Optimal);
save Optimal;
display(exploitCount);
display(exploreCount);