function T1 = build_T1(rho,N,nel)
    r=2*N;
    G = get_1RDM_from_2RDM(rho,r,nel);
    T1 = zeros(r,r,r,r,r,r);
    
    
    function c = get_T1_ijklmn(a,b,c,d,e,f)
    delta = @(i,j) i==j;

    g1 = [a,b,c];
    g2=[d,e,f];
    c=0;
    for i=1:3
        for j=1:3
            for k=1:3
                for l=1:3
                    for m = 1:3
                        for n=1:3
                            c=c+levicivita([i,j,k])*levicivita([l,m,n])*...
                                (...
                                1/6*delta(g1(i),g2(l))*delta(g1(j),g2(m))*delta(g1(k),g2(n)) ...
                                -  0.5*delta(g1(i),g2(l))*delta(g1(j),g2(m))*G(g1(k),g2(n)) +...
                                0.5*delta(g1(i),g2(l))* get_rho_from_matrix(rho,g1(j),g1(k),g2(m),g2(n),r) ...
                            );
                 
                        end
                    end
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
                            T1(ii,jj,kk,ll,mm,nn)=get_T1_ijklmn(ii,jj,kk,ll,mm,nn);
                        end
                    end
                end
            end
        end
    end
end

