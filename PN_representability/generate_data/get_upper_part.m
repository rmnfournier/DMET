function [L] = get_upper_part(R)
%get_upper part of R, L has its elements ordered by columns of R

    % R is positive defined, but due to round-off errors, it may have
    % negative eigen values close to 0, we need to regularize the matrix
    min_eig_v = min(eig(R));

    % Get the Cholevski decomposition
    R = chol(R);
    % Save only the upper triangular part
    index = triu(ones(size(R)));
    L = R(index==1);
    L=L(:);
end

