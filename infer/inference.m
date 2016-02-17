function[n_correct]=inference(gold_tree,parameter,batch)
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,0);
    Span=ConstructTree(batch,h_edu{1},parameter);
    n_correct=length(find((Span-gold_tree.Span)==0));
    [n_correct/length(gold_tree.AllNodes),length(gold_tree.AllNodes)]
end
    
function[Span]=ConstructTree(batch,h_edu,parameter)
    CurrentNodes=[];
    for i=1:size(h_edu,2)
        node=Tree_Node();
        node.h=h_edu(:,i);
        node.c=zeroMatrix([parameter.dimension,1]);
        node.isleaf=1;
        node.leaf_index=i;
        node.span=[i,i];
        CurrentNodes=[CurrentNodes,node];
    end
    Span=eye(length(CurrentNodes));
    concate=[];
    for i=1:size(h_edu,2)-1
        concate=[concate,[h_edu(:,i);h_edu(:,i+1)]];
        scores=parameter.nonlinear_gate_f(parameter.U*parameter.nonlinear_f(parameter.W*concate));
    end
    while 1==1
        [max_score,max_index]=max(scores);
        node=Tree_Node();
        child1=CurrentNodes(max_index);
        child2=CurrentNodes(max_index+1);
        node.children{1}=child1;
        node.children{2}=child2;
        node.span=[child1.span(1),child2.span(2)];
        Span(child1.span(1),child2.span(2))=1;
        child1.parent=node;
        child2.parent=node;
        lstmUnit_tree(parameter,node,node.children{1}.h,node.children{2}.h,0);
        CurrentNodes(max_index)=node;
        CurrentNodes(max_index+1)=[];
        if max_index~=1
            concate=[CurrentNodes(max_index-1).h;node.h];
            scores(max_index-1)=parameter.nonlinear_gate_f(parameter.U*parameter.nonlinear_f(parameter.W*concate));
        end
        if max_index~=length(scores)
            concate=[node.h;CurrentNodes(max_index+1).h];
            scores(max_index)=parameter.nonlinear_gate_f(parameter.U*parameter.nonlinear_f(parameter.W*concate));
            scores(max_index+1)=[];
        else 
            scores(max_index)=[];
        end
        if length(CurrentNodes)==1
            root_node=CurrentNodes(1);
            break;
        end
    end
end

