function Q = build_Q(rho,N,nel)
    r=2*N;
    G = get_1RDM_from_2RDM(rho,r,nel);
    Q = zeros(r,r,r,r);
    d = @(i,j) i==j;
    for i = 1:r
        for j=1:r
            for k=1:r
                for l=1:r
                    % define the matrix coordonate
                    Q(i,j,k,l)=(d(i,k)*d(j,l)-d(i,l)*d(j,k)) -(d(i,k)*G(j,l)+d(j,l)*G(i,k)) +(d(i,l)*G(j,k)+d(j,k)*G(i,l))-2*get_rho_from_matrix(rho,i,j,k,l,r);
                end
            end
        end
    end
end