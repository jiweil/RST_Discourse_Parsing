function[grad]=Backward_Tree(tree,grad,parameter)
    grad.tree_W=zeroMatrix(size(parameter.tree_W));
    grad=back_tree(tree.root,parameter,grad);
    grad.tree_W=grad.tree_W/size(tree.clique_vector,1)+parameter.C*parameter.tree_W;
    grad.leaves=[];
    for i=1:length(tree.LeafNodes)
        grad.leaves=[grad.leaves,tree.LeafNodes{i}.dh];
    end
end

function[grad]=back_tree(node,parameter,grad)
    if node.nuclear==1
        grad=back_tree(node.children{1},parameter,grad);
        grad=back_tree(node.children{2},parameter,grad);
    elseif node.isleaf==1
        return;
    else
        zeroState=zeroMatrix([parameter.dimension,1]);
        if length(node.dc)==0
            node.dc=zeroState;
        end
        if length(node.dh)==0
            node.dh=zeroState;
        end
        node.dc=arrayfun(@plusTanhPrimeTriple,node.dc,node.lstm.f_c_t,node.lstm.o_gate,node.dh);
        do=arrayfun(@sigmoidPrimeTriple,node.lstm.o_gate,node.lstm.f_c_t,node.dh);
        di=arrayfun(@sigmoidPrimeTriple,node.lstm.i_gate,node.lstm.a_signal,node.dc);
        df_l=arrayfun(@sigmoidPrimeTriple,node.lstm.f_l_gate,node.children{1}.c,node.dc);
        df_r=arrayfun(@sigmoidPrimeTriple,node.lstm.f_r_gate,node.children{2}.c,node.dc);
        if length(node.children{1}.dc)==0
            node.children{1}.dc=node.lstm.f_l_gate.*node.dc;
        else 
            node.children{1}.dc=node.children{1}.dc+node.lstm.f_l_gate.*node.dc;
        end
        if length(node.children{2}.dc)==0
            node.children{2}.dc=node.lstm.f_r_gate.*node.dc;
        else
            node.children{2}.dc=node.children{2}.dc+node.lstm.f_r_gate.*node.dc;
        end

        dl=arrayfun(@tanhPrimeTriple,node.lstm.a_signal,node.lstm.i_gate,node.dc);
        d_ifoa=[di;df_l;df_r;do;dl];
        grad.tree_W=grad.tree_W+d_ifoa*node.lstm.input';
        g_input=parameter.tree_W'*d_ifoa;
        if parameter.dropout~=0
            g_input=g_input.*node.lstm.drop_left;
        end
        if length(node.children{1}.dh)==0
            node.children{1}.dh=g_input(1:parameter.dimension,:);
        else 
            node.children{1}.dh=node.children{1}.dh+g_input(1:parameter.dimension,:);
        end
        if length(node.children{2}.dh)==0
            node.children{2}.dh=g_input(parameter.dimension+1:2*parameter.dimension,:);
        else 
            node.children{2}.dh=node.children{2}.dh+g_input(parameter.dimension+1:2*parameter.dimension,:);
        end
        grad=back_tree(node.children{1},parameter,grad);
        grad=back_tree(node.children{2},parameter,grad);
    end
end


