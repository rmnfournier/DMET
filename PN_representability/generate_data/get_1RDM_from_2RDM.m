function [rho_1]= get_1RDM_from_2RDM(rho,r,nel)
% return the rxr 1-RDM corresponding to the 2-RDM rho
    rho_1 = zeros(r);
    for i = 1 : r
        for j = 1:r
            for k = 1:r
                rho_1(i,j)=rho_1(i,j)+get_rho_from_matrix(rho,i,k,j,k,r);
            end
        end
    end
    rho_1=rho_1*2/(nel-1);
end