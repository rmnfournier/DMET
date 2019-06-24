function GG = build_G(rho,N,nel)
    r=2*N;
    G = get_1RDM_from_2RDM(rho,r,nel);
    GG = zeros(r,r,r,r);
    d = @(i,j) i==j;
    for i = 1:r
        for j=1:r
            for k=1:r
                for l=1:r
                    GG(i,j,k,l)= d(j,l)*G(i,k)-2*get_rho_from_matrix(rho,i,l,k,j,r);
                end
            end
        end
    end
end