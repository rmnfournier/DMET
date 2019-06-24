function [Energy] = get_rho_energy(rho,u)
    t=1;
    rho1=get_1RDM_from_2RDM(rho,12,6);
    Ecin=0;
    list_bin = [[1,2];[2,4];[4,6];[6,5];[5,3];[3,1]]';
    for bin = list_bin
        i = bin(1);
        j= bin(2);
       Ecin = Ecin -t* (rho1( i,j)+rho1( j,i)+rho1(i+6,j+6)+rho1( j+6,i+6));
    end
    Epot=0;
    for i = 1:6
        Epot = Epot + u * get_rho_from_matrix(rho,i,i+6,i,i+6,12);
    end
    
    Energy = Ecin+Epot;
end

