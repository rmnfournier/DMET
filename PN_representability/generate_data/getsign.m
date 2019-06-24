function [s] = getsign(state)
%Get relative phase
hex_order = [1,2,4,6,5,3,7,8,10,12,11,9];
% keep track of the current order
order = state .* (1:12);
% then, put it in the hex order
state_in_hex = order(hex_order);
%keep the non-0 elements
state_in_hex = state_in_hex(state_in_hex>0.1);
% get the ordered indices
[~,order] = sort(state_in_hex);

I=speye(length(order));
s = sign(det(I(:,order)));
end

