function s = levicivita(p)
    I = eye(length(p));
    s = det(I(:,p));
end