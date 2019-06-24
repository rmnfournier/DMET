%% Get the 2RDM of some examples 


syms u
syms t

H=[u,0, -t,t,0,0;0,0,0,0,0,0;-t,0,0,0,0,-t;t,0,0,0,0,t;0,0,0,0,0,0;0,0,-t,t,0,u];
[v,w] = eig(H);
w=diag(w);


list_u = [0,1,2,4,6,8];
c_t=1;
f=figure();
c=0;
list_en = [];
for c_u = list_u
   c=c+1;
    %% get the energy and the ground state
    [e,arg_psi] = min(eval(subs(subs(w,u,c_u),t,c_t)));
    psi = eval(subs(subs(v(:,arg_psi),u,c_u),t,c_t));
    psi=psi/norm(psi);
    a_psi=abs(psi);
    %% plot
    r2 = psi_to_2RDM(psi,2,2);
    r = get_1RDM_from_2RDM(r2,4,2);
    list_en=[list_en e];
    H1=-2*(a_psi(1)+a_psi(6))*(a_psi(3)+a_psi(4));
    H2=c_u*(a_psi(1)^2+a_psi(6)^2);
    '-----';
    subplot(2,length(list_u),c)
    heatmap(r);
    subplot(2,length(list_u),c+length(list_u))
    heatmap(r2);
end
list_en