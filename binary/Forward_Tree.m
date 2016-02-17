function[h]=Forward_Tree(node,parameter,h_text,isTraining)
    if node.isleaf==1
        node.h=h_text(:,node.sen_index);
        node.c=zeroMatrix([parameter.dimension,1]);
        h=node.h;
    else
        h_left=Forward_Tree(node.children{1},parameter,h_text,isTraining);
        h_right=Forward_Tree(node.children{2},parameter,h_text,isTraining);
        h=lstmUnit_tree(parameter,node,h_left,h_right,isTraining);
    end
end
