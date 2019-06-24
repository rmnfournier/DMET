function [index] = get_matrix_coef(i,j,r)
%get_matrix_coef help to represent a 4th order tensor into a 2D matrix
%   i,j are two coefficients of the forth order tensor (R_ijkl)
%   r is the number of spinsites available for an electrons 
index = j-i+(2*r-i)*(i-1)/2;
end