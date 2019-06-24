%% By Romain Fournier
% 03.05.2019
% Generate 2 particles-Reduced Density Matrix (2-RDM)

%% Parameters 
tic
close all
clear all

foldername ='../data_L6_N6_super_sym'; % name of the folder in which the data will be saved
filename = 'data.csv' ; % name of the file in which the data will be saved
sym=true ; % if sym -> spin un = spin down 

L = 6 ; % Number of sites 
N = 6 ; % Number of electrons 

nb_data = 10000; % Number of Matrix that must be saved
[J_up,J_down,J]=build_look_up_table(L,N);

get_state = @(v) sum(2.^(0:(2*L-1)).*v);
%% Simulation
dimension = nchoosek(2*L,N); % we have 2L possible location for the electrons (spin up and down), and we place N electrons
h=waitbar(0,'generating data');
%% Prepare the list of non equivalent states
list_states = [];
list_sym_states = [];
index=[];
if sym &&  N==6&&L==6
    for runner = 1:dimension
        psi_int = J(runner); % get the current state
        if(sum(bitget(psi_int,1:6))==3)
            % Check if it is in the list
            if ~(ismember(psi_int,list_states))
                v = bitget(psi_int,1:12); % convert to array
                states_with_sym = get_sym_states(v,J); % get all states with the same symmetry
                index = [index,states_with_sym];
                list_states = [list_states,J(states_with_sym)'];
                list_sym_states = [list_sym_states, psi_int];
            end
        end
    end
end
for ii = 1:nb_data
    %% Generate a wave vector
    waitbar(ii/nb_data,h);
    if sym 
        psi = zeros(dimension,1);
        if N==2 && L==2
            coef = normrnd(0,1,3,1); % todo generalize of other than N=L=2
            psi(1)=coef(1);
            psi(6)=coef(1);
            psi(2)=coef(2);
            psi(5)=coef(2);
            psi(3)=coef(3);
            psi(4)=-coef(3);
        elseif N==6&&L==6
            for psi_int = list_sym_states
                v = bitget(psi_int,1:12); % convert to array
                states_with_sym = get_sym_states(v,J); % get all states with the same symmetry
                coef = normrnd(0,1); % generate random number
                psi(states_with_sym)=coef; % set the same coefficient to all states with the same symmetry
            end
           % check the sign
           for runner = 1:dimension
                psi_int = J(runner); % get the current state
                v = bitget(psi_int,1:12); % convert to array
                s = getsign(v);
                psi(runner)=psi(runner)*s;
           end
        else
            error("sym only supprots N=L=2 and N=L=6")
        end
    else
        psi = normrnd(0,1,dimension,1); % Random normal vector
    end
    psi = psi/norm(psi); % Normalize it

    %%  Get the density Matrix
    rho = psi_to_2RDM(psi,L,N); % Rho is a 2D matrix(Rho ijkl = rho j-i+(2r-i)/2,l-k+(2r-k)(k-1)/2)

    %% Save the upper part 
    rho_up = get_upper_part(rho); % L is a vector containing the upper part of rho (since rho ij,kl = rho* kl,ij)
    
    %% Save the result
    dlmwrite(foldername+"/"+filename,rho_up','-append','delimiter',',');
end
close(h);

toc
