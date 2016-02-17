function[]=Infer(Test_Text,binary,multi,RelationList,SpanList,IndexToRelation)
    span_correct=0;relation_correct=0;total_span=0;total_relation=0;
    for i=1:length(RelationList)
        batch=Test_Text{i};
        gold_span=SpanList{i};
        gold_relation=RelationList{i};
        [InferSpan,InferRelation]=inference_span(binary,multi,batch,IndexToRelation);
        span_correct=span_correct+length(find((gold_span-InferSpan)==0));
        relation_correct=relation_correct+length(find((gold_relation-InferRelation)==0));
        total_span=total_span+length(find(gold_span~=100));
        total_relation=total_relation+length(find(gold_relation~=100));

    end
    disp('span')
    span_correct/total_span
    disp('relation')
    relation_correct/total_relation
end

function[InferSpan,InferRelation]=inference_span(binary,multi,batch,IndexToRelation)
    [h_edu_binary,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,binary,0);
    [h_edu_multi,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,multi,0);
    [InferSpan,InferRelation]=ConstructTree(batch,h_edu_binary{1},h_edu_multi{1},binary,multi,IndexToRelation);
end
    
function[InferSpan,InferRelation]=ConstructTree(batch,h_edu_binary,h_edu_multi,binary,multi,IndexToRelation)
    CurrentNodes=[];
    for i=1:size(h_edu_binary,2)
        node=Tree_Node();
        node.h_binary=h_edu_binary(:,i);
        node.h_multi=h_edu_multi(:,i);
        node.c_binary=zeroMatrix([binary.dimension,1]);
        node.c_multi=zeroMatrix([binary.dimension,1]);
        node.isleaf=1;
        node.leaf_index=i;
        node.span=[i,i];
        CurrentNodes=[CurrentNodes,node];
    end
    InferSpan=eye(length(CurrentNodes));
    InferRelation=zeros(length(CurrentNodes),length(CurrentNodes));
    concate=[];
    for i=1:size(h_edu_binary,2)-1
        concate=[concate,[h_edu_binary(:,i);h_edu_binary(:,i+1)]];
        scores=binary.nonlinear_gate_f(binary.U*binary.nonlinear_f(binary.W*concate));
    end
    while 1==1
        [max_score,max_index]=max(scores);
        node=Tree_Node();
        child1=CurrentNodes(max_index);
        child2=CurrentNodes(max_index+1);
        node.children{1}=child1;
        node.children{2}=child2;
        node.span=[child1.span(1),child2.span(2)];
        InferSpan(child1.span(1),child2.span(2))=1;
        child1.parent=node;
        child2.parent=node;
        concate=[child1.h_multi;child2.h_multi];
        interme=multi.nonlinear_f(multi.W*concate);
        multi_scores=multi.U*interme;
        [value,prediciton]=max(multi_scores);
        M=IndexToRelation(prediciton,:);
        InferRelation(child1.span(1),child1.span(2))=M(1);
        InferRelation(child2.span(1),child2.span(2))=M(2);

        lstmUnit_tree(binary,node,0,1);
        lstmUnit_tree(multi,node,0,0);
        CurrentNodes(max_index)=node;
        CurrentNodes(max_index+1)=[];
        if max_index~=1
            concate=[CurrentNodes(max_index-1).h_binary;node.h_binary];
            scores(max_index-1)=binary.nonlinear_gate_f(binary.U*binary.nonlinear_f(binary.W*concate));
        end
        if max_index~=length(scores)
            concate=[node.h_binary;CurrentNodes(max_index+1).h_binary];
            scores(max_index)=binary.nonlinear_gate_f(binary.U*binary.nonlinear_f(binary.W*concate));
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

function[h]=lstmUnit_tree(parameter,node,isTraining,isBinary)%lstm unit calculation
    if isBinary==1
        h_left=node.children{1}.h_binary;
        h_right=node.children{2}.h_binary;
    else
        h_left=node.children{1}.h_multi;
        h_right=node.children{2}.h_multi;
    end
    input=[h_left;h_right];
    if parameter.dropout~=0&&isTraining==1
        if parameter.CheckGrad==1   
            drop_left=repmat(parameter.drop_left,1,size(input,2));
        else
            drop_left=randSimpleMatrix(size(input))<1-parameter.dropout;
        end
        input=input.*drop_left;
    end
    if isTraining==0
        input=input.*(1-parameter.dropout);
    end
    ifoa_linear =parameter.tree_W*input;
    ifo_gate=parameter.nonlinear_gate_f(ifoa_linear(1:4*parameter.dimension,:));
    i_gate = ifo_gate(1:parameter.dimension, :);
    f_l_gate = ifo_gate(parameter.dimension+1:2*parameter.dimension,:);
    f_r_gate = ifo_gate(parameter.dimension*2+1:3*parameter.dimension,:);
    o_gate=ifo_gate(parameter.dimension*3+1:4*parameter.dimension,:);
    a_signal = parameter.nonlinear_f(ifoa_linear(4*parameter.dimension+1:5*parameter.dimension,:));
    if isBinary==1
        c_t=f_l_gate.*node.children{1}.c_binary+f_r_gate.*node.children{2}.c_binary+i_gate.*a_signal;
    else
        c_t=f_l_gate.*node.children{1}.c_multi+f_r_gate.*node.children{2}.c_multi+i_gate.*a_signal;
    end
    f_c_t = parameter.nonlinear_f(c_t);
    h=o_gate.*f_c_t;
    if isBinary==1
        node.h_binary=h;
        node.c_binary=c_t;
        node.lstm_binary.input = input;
        node.lstm_binary.i_gate = i_gate;
        node.lstm_binary.f_l_gate = f_l_gate;
        node.lstm_binary.f_r_gate = f_r_gate;
        node.lstm_binary.o_gate=o_gate;
        node.lstm_binary.a_signal = a_signal;
        node.lstm_binary.f_c_t = f_c_t;
        if isTraining==1&&parameter.dropout~=0
            node.lstm_binary.drop_left=drop_left;
        end
    else
        node.h_multi=h;
        node.c_multi=c_t;
        node.lstm_multi.input = input;
        node.lstm_multi.i_gate = i_gate;
        node.lstm_multi.f_l_gate = f_l_gate;
        node.lstm_multi.f_r_gate = f_r_gate;
        node.lstm_multi.o_gate=o_gate;
        node.lstm_multi.a_signal = a_signal;
        node.lstm_multi.f_c_t = f_c_t;
        if isTraining==1&&parameter.dropout~=0
            node.lstm_multi.drop_left=drop_left;
        end
    end
end


function[node]=Free(node)
    node.c=[];
    node.lstm=[];
    node.h=[];
    node.dh=[];
    node.dc=[];
    for i=1:length(node.children)
        Free(node.children{i});
    end
end

function[]=testing(parameter,Test_Trees,Test_Text)
    total_cost=0;total_num=0;total_right=0;
    for i=1:length(Test_Trees)
        tree=Test_Trees{i};
        batch=Test_Text{i};
        [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,0);
        Forward_Tree(tree.root,parameter,h_edu{1},0);
        [cost,n_right]=softmax1(tree,parameter);
        total_cost=total_cost+cost;
        total_num=total_num+size(tree.clique_vector,1);
        total_right=total_right+n_right;
    end
    disp('cost')
    total_cost/total_num
    disp('accuracy')
    total_right/total_num
end
