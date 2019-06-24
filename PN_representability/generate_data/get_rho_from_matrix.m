function rho_ijkl=get_rho_from_matrix(rho,i,j,k,l,r)
%% computes rhoijkl from the matrix representation of rho
if i== j || k==l
    rho_ijkl=0;
else
    signe=1;
    if(i>j)
        signe=signe*-1;
    end
    if(k>l)
        signe=signe*-1;
    end
    x=get_matrix_coef(min(i,j),max(i,j),r);
    y=get_matrix_coef(min(k,l),max(k,l),r);
    rho_ijkl=signe*rho(x,y);
end
end