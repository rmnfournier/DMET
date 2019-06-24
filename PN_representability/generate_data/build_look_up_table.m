function [J_up,J_down,J] = build_look_up_table(L,N)
%build_look_up_table map the occupation representation of the states with N electrons on L sites with an
%index 
%   L number of sites, N number of electrons
%   J[i] gives an int which is the occupation representation of the ith
%   basis element 
%   Starting with a state i, we can get the index using the relation index
%   J_up( spin_up(i))+J_down(spin_down(i))

% Fix the number of electron with spin down
n_states_per_spin = 2^(L);
% Allocate the tables
J_up = zeros(n_states_per_spin,1);
J_down = zeros(n_states_per_spin,1);
J = zeros(nchoosek(2*L,N),1);

j_down=0; % a first runner to keep track of the index
for n_down = 0:N
    n_up = N-n_down;
    % for this configuration, we will have n_states_up possible states
    n_states_up = nchoosek(L,n_up);
    
    % Now we iterate over all states and update the lookup table if the
    % filling is correct
    j_up=0; %The second runner to keep track of the current index
    for down_part = 0 : n_states_per_spin-1
        n_down_electrons = sum(bitget(down_part,L:-1:1));
        if n_down_electrons == n_down
            for upper_part = 0 : n_states_per_spin-1
                n_up_electrons = sum(bitget(upper_part,L:-1:1));
                if n_up_electrons == n_up
                    J_up(upper_part+1)=j_up; 
                    J_down(down_part+1)=j_down;
                    J(j_up+j_down+1) = n_states_per_spin*upper_part+down_part;
                    j_up=j_up+1;
                end
            end
            j_up=0; % restart the second runner
            j_down = j_down + n_states_up; % move the second one
        end
    end
end

end

