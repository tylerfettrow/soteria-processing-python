function normedVector = normVector(vector)
    normedVector = zeros(size(vector));
    for i_col = 1 : size(vector, 2)
%         normedVector(:, i_col) = vector(:, i_col) * norm(vector(:, i_col))^(-1);
       normedVector(:, i_col) = ((sqrt(vector(:,1)^2 + vector(:,2)^2 + vector(:,3)^2))/1);
%        current_mag = sum(sqrt(vector(:,1)^2 + vector(:,2)^2 + vector(:,3)^2); 
       
    end


end

