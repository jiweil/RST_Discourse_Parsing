function[grad,cost,prediciton]=softmax(tree,parameter)
    concate=[];
    for i=1:size(tree.clique_vector,1)
        concate=[concate,[tree.AllNodes{tree.clique_vector(i,1)}.h;tree.AllNodes{tree.clique_vector(i,2)}.h]];
    end
    interme=parameter.nonlinear_f(parameter.W*concate);
    
    scores=parameter.U*interme;
    scores=exp(scores);
    norms = sum(scores, 1);
    scores=bsxfun(@rdivide, scores, norms);
    [asdfasf,prediciton]=max(scores);
    
    scoreIndices = sub2ind(size(scores),tree.clique_vector(:,3)',1:size(scores,2) );
    cost=sum(-log(scores(scoreIndices)));
    scores(scoreIndices) =scores(scoreIndices) - 1;

    grad.U=scores*interme';

    d_in=arrayfun(@tanhPrime,interme,parameter.U'*scores);

    grad.W=d_in*concate';
    d_h=parameter.W'*d_in;
    for i=1:size(tree.clique_vector,1)
        if length(tree.AllNodes{tree.clique_vector(i,1)}.dh)==0
            tree.AllNodes{tree.clique_vector(i,1)}.dh=d_h(1:parameter.dimension,i);
        else tree.AllNodes{tree.clique_vector(i,1)}.dh=tree.AllNodes{tree.clique_vector(i,1)}.dh+d_h(1:parameter.dimension,i);
        end
        if length(tree.AllNodes{tree.clique_vector(i,2)}.dh)==0
            tree.AllNodes{tree.clique_vector(i,2)}.dh=d_h(parameter.dimension+1:2*parameter.dimension,i);
        else tree.AllNodes{tree.clique_vector(i,2)}.dh=tree.AllNodes{tree.clique_vector(i,2)}.dh+d_h(parameter.dimension+1:2*parameter.dimension,i);
        end
    end
    if parameter.CheckGrad==1
        cost=cost/size(tree.clique_vector,1);
    end
    grad.U=grad.U/size(tree.clique_vector,1);
    grad.W=grad.W/size(tree.clique_vector,1)+parameter.C*parameter.W;
end

