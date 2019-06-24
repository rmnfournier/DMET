function [rho] = psi_to_2RDM(psi,L,N)
%psi_to_2RDM Get the 2 particles reduced density matrix
%   psi is the wavefunction, L the number of sites, N the number of
%   electrons

    %% Build the lookup table if necessary for the given N and L 
    persistent J;
    persistent J_up;
    persistent J_down;
    
    if isempty(J)
        [J_up,J_down,J]=build_look_up_table(L,N);
    end
    
    %% Define a nested function to help to compute rho_ijkl
    function [c] = rho_ijkl(i,j,k,l)
        %rho_ijkl computes <psi|c+ic+jclck|psi>
        c=0;
        bitflip = @(val, n) bitxor(val, 2.^(n-1));
        check_sign = @(state,k) (-1)^sum(bitget(state,(k-1):-1:1));
        %% iterate over all states
        for ii = 1:length(psi)
            %% check if the jump is not null
              state = J(ii); % get the current state in occupation representation
              signe=1; % keep track of changes in sign 
              if all(bitget(state,[k,l])) && k~=l % check if kth and lth positions are occupied and k differs from l
                  tmp_state = bitflip(state,k);
                  signe = signe * check_sign(state,k);
                  tmp_state = bitflip(tmp_state,l);
                  signe = signe * check_sign(tmp_state,l);

                  if all(~bitget(tmp_state,[i,j])) && i~=j % check if ith and jth positions are empty 
                 %% Get the target state
                    tmp_state = bitflip(tmp_state,j);
                    signe = signe * check_sign(tmp_state,j);
                    tmp_state = bitflip(tmp_state,i);
                    signe = signe * check_sign(tmp_state,i);
                    
                 %% Get the index of the final state
                    new_index = J_down(rem(tmp_state,2^L)+1)+J_up(fix(tmp_state/2^L)+1)+1;
                    %new_index=J_down(1+sum(bitget(tmp_state,1:L)))+J_up(1+sum(bitget(tmp_state,(L+1):2*L)))+1;
                 %% update c
                    c = c+signe*psi(ii)*conj(psi(new_index));
                  end
              end


        end

    end
    %% Compute the size of rho 
    r = 2*L; % number of positions for an electron
    d = r*(r-1)/2;
    %% iterate to get the coefficients 
    rho = zeros(d);
    for i = 1:r
        for j = i+1:r
            for k = 1:r
                for l = k+1:r
                    x = get_matrix_coef(i,j,r);
                    y = get_matrix_coef(k,l,r);
                    rho(x,y)=rho_ijkl(i,j,k,l);
                end
            end
        end
    end
    min_eig=min(eig(rho));
    if(min_eig<1e-15)
        rho = rho+diag(diag(ones(size(rho))))*(abs(min_eig)*10+1e-9);
    end
    rho=rho/2;
end

