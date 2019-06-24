function T2 = build_T2(rho,N,nel)
    r=2*N;
    G = get_1RDM_from_2RDM(rho,r,nel);
    T2 = zeros(r,r,r,r,r,r);
    
    
    function c = get_T1_ijklmn(i,b,c,l,e,f)
    delta = @(i,j) i==j;

    g1 = [b,c];
    g2=[e,f];
    c=0;
        for j=1:2
            for k=1:2
                    for m = 1:2
                        for n=1:2
                            c=c+levicivita([j,k])*levicivita([m,n])*...
                                (...
                                 0.5*delta(g1(j),g2(m))*delta(g1(k),g2(n))*G(i,l)  ...
                                 +0.5*delta(i,l)*get_rho_from_matrix(rho,g1(m),g1(n),g2(j),g2(k),r) ...
                                 -2*delta(g1(j),g2(m))*get_rho_from_matrix(rho,i,g1(n),l,g2(k),r)... 
                            );
                 
                        end
                    end
                
            end
        end
    
    
end
    
    for ii = 1:r
        for jj=1:r
            for kk=1:r
                for ll=1:r
                    for mm=1:r
                        for nn=1:r
                            T2(ii,jj,kk,ll,mm,nn)=get_T1_ijklmn(ii,jj,kk,ll,mm,nn);
                        end
                    end
                end
            end
        end
    end
end

