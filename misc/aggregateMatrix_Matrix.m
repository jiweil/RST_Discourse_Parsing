function [accumX, uniqIndices] = aggregateMatrix_Matrix(Y, indices)

    [uniqIndices, ~, J] = unique(indices);
    numUniqIndices = length(uniqIndices);
    numEmbGrads = length(indices);
    X=reshape(Y,size(Y,1)*size(Y,2),size(Y,3));

    if numEmbGrads==1
        accumX = X;
    else
        sparseMatrix = zeros(numEmbGrads, numUniqIndices,'double', 'gpuArray');
        sparseIndices = sub2ind([numEmbGrads, numUniqIndices], 1:numEmbGrads, J'); 
        sparseMatrix(sparseIndices) = ones(numEmbGrads, 1);
        accumX = X*sparseMatrix;
    end
    accumX=reshape(accumX,size(Y,1),size(Y,2),numUniqIndices);
end
