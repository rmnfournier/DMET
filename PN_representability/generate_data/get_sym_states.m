function [index_list_states] = get_sym_states(v,J)
%get_sym_states get the sym states for the ring configuration
%   J is the lookup table
%   v contains 6 1 and 0 

%initialize empty list 
index_list_states=[]; 
get_int_state = @(v) sum(2.^(0:11).*v);
get_index = @(int_state) find(J==int_state);

% rotation sym 
rotations = [3,1,5,2,6,4,9,7,11,8,12,10];
% mirrors
o = [2,1,4,3,6,5,8,7,10,9,12,11];
% spin inversion
n = [7:12,1:6];
% both
on = o(n);


for i = 1:6
    iv = 1-v;
    
    index = get_index(get_int_state(v));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(v(n)));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(v(o)));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(v(on)));
    index_list_states=[index_list_states, index];

    index = get_index(get_int_state(iv));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(iv(n)));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(iv(o)));
    index_list_states=[index_list_states, index];
    
    index = get_index(get_int_state(iv(on)));
    index_list_states=[index_list_states, index];


    v = v(rotations);
end
index_list_states=unique(index_list_states);
end

