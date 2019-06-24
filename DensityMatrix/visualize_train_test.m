%% Set up the hamiltonian to get analytic results
us = [0,1,2,3,4,5];
t=1;
syms s_u
syms s_t

H=[s_u,0, -s_t,s_t,0,0;0,0,0,0,0,0;-t,0,0,0,0,-t;t,0,0,0,0,t;0,0,0,0,0,0;0,0,-s_t,s_t,0,s_u];
[v,w] = eig(H);
w=diag(w);


ms= ["50","75","100"];
nb_data = ["1000","2000","5000"];
l_ms = length(ms);
l_nb=length(nb_data);
energy = zeros(l_ms,l_nb);
std_energy = zeros(l_ms,l_nb);


c=0;
c_m=0;
runner_u=0;
results=zeros(length(us),length(nb_data));
for u =us
    runner_u=runner_u+1;
    c_m=c_m+1;
    e0= min(eval(subs(subs(w,s_u,u),s_t,t)))
    for m = ms
        c_n=0;
        for n =nb_data
            c_n=c_n+1;
            c=c+1;
             loss = importdata("loss_"+m+"_"+n+"_episodes_1000.csv");
             figure(1)
             subplot(l_ms,l_nb,c)
             plot(loss.data(:,[2,3]))
             xlabel("epochs")
             ylabel("Cross entropy")
             title("m="+m+", nb\_data="+n)
%             energies=importdata("energies_u_"+num2str(u)+"_"+m+"_"+n+".csv");
%             energies=energies.data(:,[2,3]);
%             energies=energies(:);
%             energy(c_m,c_n)=min(energies);
%             std_energy(c_m,c_n)=std(energies);
%             results(runner_u,c_n)=(min(energies)-e0)/e0;
        end
        f=figure(c_n+2);
       % subplot(1,length(us),c_m)
        %h1=plot(str2double(nb_data),(energy(c_m,:)-e0)/abs(e0),markers{runner_u});
       % set(h1, 'markerfacecolor', get(h1, 'color')); 
      %  hold on
        %plot(str2double(nb_data),repmat(analytic(c_m),length(nb_data),1),'-')
        %xlabel("Dataset size")
       % ylabel("$\frac{E_{gan}-E_{th}}{|E_{th}|}$",'Interpreter','latex')
      %  set(gca,"xscale","log")
      %  legend("U=0","U=t","U=2t","U=3t","U=4t")
        %legend('Predicted','Analytic')
        %axis([900,1.1*10^4,-2.15,0])
%         %print_figure(f,"pn_representability_convergence_2N_2L_sym.eps",8.6,8.6)
        
    end
end

% print_figure(f,"pn_representability_gan_E_N_2_L_2.eps",17.2,8.6)


%% Energies
% markers = {'--o','--s','--d','--^','--v','--h'};
% runner_m=0;
% ms = 10;
% us = 0:1;
% nb_data = ["5000"];
% e_0s=[-8];
% ms=50;
% for m =ms
%     results = zeros(length(us),length(nb_data));
%     runner_m=runner_m+1;
%     runner_u=0;
%     for u = us
%         runner_u=runner_u+1;
%         runner_n=0;
%         e0=e_0s(runner_u);
%         for n =nb_data
%             runner_n =  runner_n+1;
%             energies=importdata("energies_u_"+num2str(u)+"_"+m+"_"+n+".csv");
%             energies=energies.data(:,[2,3]);
%             energies=energies(:);
%             results(runner_u,runner_n)=(min(energies)-e0)/abs(e0);
%         end
%         f=figure(1);
%         subplot(1,length(ms),runner_m)
%         h1=plot(str2double(nb_data),results(runner_u,:),markers{runner_u});
%         %set(h1, 'markerfacecolor', get(h1, 'color')); 
%         hold on
%     end
%     title("m="+num2str(m))
%     legend("U=0","U=t","U=2t","U=3t","U=4t","U=5t")
%     xlabel("Dataset size")
%     ylabel("$\frac{E_{gan}-E_{th}}{|E_{th}|}$",'Interpreter','latex')
%     set(gca,"xscale","log")
%     %axis([900,10000,0,10])
% end
% 
% 
% 
% 
% 
% 
% 
% 



















