function [pass,failed]=check_rho(rho,N,nel)
%check_rho : return True if rho passes the P Q G conditions from Nakamara
%paper. N is the number of sites
% rho is a potential 2-RDM 

pass =true;
failed=0;
check_pos_dev_matrix = @(M) min(real(eig(M)))>-1e-7;
check_pos_dev_tensor = @(M) min(heig(M))>-1e-7;
r=2*N;

%% trace 1-RDM
if pass
    rho_1 = get_1RDM_from_2RDM(rho,r,nel);
    pass = abs(nel-trace(rho_1))<1e-6;
    if ~pass
        failed=1;
        "trace rho_1 : "+int2str(pass)
    end
end
%% partial trace
if pass
    somme=0;
    for i=1:r
        for j = 1:r
            somme=somme+get_rho_from_matrix(rho,i,j,i,j,r);
        end
    end
    pass = abs(somme-nel*(nel-1)/2)<1e-5;
    if ~pass
        failed=2;
        " partial trace : "+int2str(pass)
    end
end
%% P condition 
if pass 
    pass = check_pos_dev_matrix(rho);
    if ~pass
        failed=3;
        "P : "+int2str(pass)
    end
end
% compute the one density matrix
%% Q condition 
if pass
    p=zeros(4^r,1);
  
    Q = build_Q(rho,N,nel);
    % construct the polynomial 
    for i=1:r
    for j=1:r
    for k=1:r
    for l=1:r
        x_order =@(i,j,k,l,m) (i==m)+(j==m)+(k==m)+(l==m);
        p(r^3*x_order(i,j,k,l,1)+r^2*x_order(i,j,k,l,2) +r^1*x_order(i,j,k,l,3)+x_order(i,j,k,l,4))=p(r^3*x_order(i,j,k,l,1)+r^2*x_order(i,j,k,l,2) +r^1*x_order(i,j,k,l,3)+x_order(i,j,k,l,4))+Q(i,j,k,l);
    end
    end
    end
    end
    % heig does not work for the null polynomial
    if norm(p)>1e-5
        pass = check_pos_dev_tensor(Q);
    end
    if ~pass
     failed=4;
    "Q "+ int2str(pass)
    end
end
% %% G condition
if pass 
    p=zeros(4^r,1);
    G  = build_G(rho,N,nel);

    for i=1:r
    for j=1:r
    for k=1:r
    for l=1:r
        x_order =@(i,j,k,l,m) (i==m)+(j==m)+(k==m)+(l==m);
        p(r^3*x_order(i,j,k,l,1)+r^2*x_order(i,j,k,l,2) +r^1*x_order(i,j,k,l,3)+x_order(i,j,k,l,4))=p(r^3*x_order(i,j,k,l,1)+r^2*x_order(i,j,k,l,2) +r^1*x_order(i,j,k,l,3)+x_order(i,j,k,l,4))+G(i,j,k,l);
    end
    end
    end
    end
    % heig does not work for the null polynomial
    if norm(p)>1e-5
        pass = check_pos_dev_tensor(G);
    end
    if ~pass
        failed=5;
    "G : "+ int2str(pass)
    end

end
%% T1 Condition
if pass
    T1 = build_T1(rho,N,nel);
    p=zeros(6^6,1);
    for i=1:r
    for j=1:r
    for k=1:r
    for l=1:r
    for m = 1:r
    for n=1:r
        x_order =@(i,j,k,l,m,n,o) (i==o)+(j==o)+(k==o)+(l==o)+(m==o)+(n==o);
        p(6^5*x_order(i,j,k,l,m,n,1)+6^4*x_order(i,j,k,l,m,n,2) +6^3*x_order(i,j,k,l,m,n,3)+6^2*x_order(i,j,k,l,m,n,4)+6*x_order(i,j,k,l,m,n,5)+x_order(i,j,k,l,m,n,6))=p(6^5*x_order(i,j,k,l,m,n,1)+6^4*x_order(i,j,k,l,m,n,2) +6^3*x_order(i,j,k,l,m,n,3)+6^2*x_order(i,j,k,l,m,n,4)+6*x_order(i,j,k,l,m,n,5)+x_order(i,j,k,l,m,n,6))+T1(i,j,k,l,m,n);
    end
    end
    end
    end
    end
    end
    
    if(norm(p)>1e-5)
    pass = check_pos_dev_tensor(T1);
    end
    if ~pass
        failed=6;
    "T1 : "+ int2str(pass)
    end
end
% %% T2 Condition
if pass
    T2 = build_T2(rho,N,nel);
    p=zeros(6^6,1);
    for i=1:r
    for j=1:r
    for k=1:r
    for l=1:r
    for m = 1:r
    for n=1:r
        x_order =@(i,j,k,l,m,n,o) (i==o)+(j==o)+(k==o)+(l==o)+(m==o)+(n==o);
        p(6^5*x_order(i,j,k,l,m,n,1)+6^4*x_order(i,j,k,l,m,n,2) +6^3*x_order(i,j,k,l,m,n,3)+6^2*x_order(i,j,k,l,m,n,4)+6*x_order(i,j,k,l,m,n,5)+x_order(i,j,k,l,m,n,6))=p(6^5*x_order(i,j,k,l,m,n,1)+6^4*x_order(i,j,k,l,m,n,2) +6^3*x_order(i,j,k,l,m,n,3)+6^2*x_order(i,j,k,l,m,n,4)+6*x_order(i,j,k,l,m,n,5)+x_order(i,j,k,l,m,n,6))+T2(i,j,k,l,m,n);
    end
    end
    end
    end
    end
    end
    
    if(norm(p)>1e-5)
    pass = check_pos_dev_tensor(T2);
    end
    if ~pass
        failed=7;
        min(heig(T2))
        "T2 : "+ int2str(pass)
    end
end


end