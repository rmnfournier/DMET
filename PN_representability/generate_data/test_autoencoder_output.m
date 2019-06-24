%% test the model
w = warning ('off','all');

somme = 0;
fail = zeros(7,1);
data = importdata("../GANs/L.csv");

for i = 1 : 50
    i
    rho=get_rho_from_data(data(i,:));
    rho1 = get_1RDM_from_2RDM(rho,4,2);
    rho=rho/trace(rho1)*2;
    
    [p,f]=check_rho(rho,2,2);
    somme=somme+p;
    if abs(p)<1e-5
        fail(f)=fail(f)+1
    end
end

function rho = get_rho_from_data(d)
    r = (-1 + sqrt(1+8*length(d)))/2;
    L=zeros(r);
    c=1;
    for j = 1:r
        for i = 1:r
            if(j>=i)
              L(i,j)=d(c) ;
              c=c+1;
            end
        end
    end
    rho=L'*L;
end