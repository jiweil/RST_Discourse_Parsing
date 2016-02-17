function[Trees]=ReadTrees(filename,isTraining)
    fd=fopen(filename);
    Trees={};
    tline = fgets(fd);
    i=0;
    while ischar(tline)
        tree=Tree();
        i=i+1;
        [tree.LeafNodes,tree.root]=ReadTree(tline);
        tree.AllNodes={};
        tree.AllNodes=GetAllNode(tree.root,tree.AllNodes);
        if isTraining==1
            tree.clique_vector=GetPositive(tree.root,tree.clique_vector);
            tree.clique_vector=GetNegative(tree.root,tree.clique_vector,tree.LeafNodes,tree.AllNodes);
            tree=TwoChild(tree.root,tree);
        end
        tree.Span=100*ones(length(tree.LeafNodes),length(tree.LeafNodes));
        GetSpan(tree,tree.root);
        %printSpan(tree.root)
        Trees{i}=tree;
        %Print(Trees{i});
        tline = fgets(fd);
    end
end

function[]=printSpan(node)
    node.index
    node.span
    for i=1:length(node.children)
        printSpan(node.children{i});
    end
end

function[span]=GetSpan(tree,node)
    if node.isleaf==1
        tree.Span(node.leaf_index,node.leaf_index)=1;
        span=[node.leaf_index,node.leaf_index];
    else
        for i=1:length(node.children)
            span=GetSpan(tree,node.children{i});
            if i==1 left=span(1);
            end
            right=span(2);
        end
        span=[left,right];
        tree.Span(left,right)=1;
    end
end

function[LeafNodes,head]=ReadTree(text)
    LeafNodes={};
    text=deblank(text);
    stack={};
    i=1;
    while(i<=length(text))
        if(text(i)=='(')
            Value='';
            i=i+1;
            node=Tree_Node();
            while(text(i)~='('&text(i)~=')')
                Value=[Value,text(i)];
                i=i+1;
            end
            space_index=strfind(Value,' ');
            if(size(space_index)==0)
                head=node;
                stack{length(stack)+1}=node;
                node.nuclear=str2num(Value(1:end));
                continue;
            end
            node.nuclear=str2num(Value(1:space_index(1)-1));
            if size(space_index,2)>0
                if size(space_index,2)==2
                    End=space_index(2)-1;
                else 
                    End=length(Value);
                end
                node.relation=str2num(Value(space_index(1)+1:End));
            end
            if size(space_index,2)==2
                node.isleaf=1;
                LeafNodes{length(LeafNodes)+1}=node;
                node.leaf_index=length(LeafNodes);
                node.sen_index=str2num(Value(space_index(2)+1:end));
            else 
                node.isleaf=0;
            end
            if(length(stack)~=0)
                node.parent=stack{length(stack)};
                num_children=length(stack{length(stack)}.children);
                stack{length(stack)}.children{num_children+1}=node;
            end
            stack{length(stack)+1}=node;
        else
            stack(length(stack))=[];
            i=i+1;
        end
    end
end

function[tree]=TwoChild(node,tree)
    if length(node.children)>2
        left_most_leaf=get_left_most_left(node);
        negatives=GetNegativeOneNode(left_most_leaf,tree.LeafNodes);
        for i=2:length(node.children)
            if i==2 left=node.children{1};
            else left=store_node;
            end
            right=node.children{i};
            if i==length(node.children)
                parent_node=node;
            else 
                parent_node=Tree_Node();
                if(left.nuclear==right.nuclear)
                    parent_node.nuclear=left.nuclear;
                end
                if(left.relation==right.relation)
                    parent_node.relation=left.relation;
                end
                parent_node.index=length(tree.AllNodes)+1;
                tree.AllNodes{length(tree.AllNodes)+1}=parent_node;
                for j=i+1:length(node.children)
                    silbing_node=node.children{j};
                    tree.clique_vector=[tree.clique_vector;[parent_node.index,silbing_node.index,1]];
                end
                for j=1:length(negatives)
                    tree.clique_vector=[tree.clique_vector;[parent_node.index,negatives(j),0]];
                end
            end
            parent_node.children={};
            parent_node.children{1}=left;
            parent_node.children{2}=right;
            left.parent=parent_node;
            right.parent=parent_node;
            store_node=parent_node;
        end
        TwoChild(node,tree);
    else
        for i=1:length(node.children)
            TwoChild(node.children{i},tree);
        end
    end
end

function[left_most_node]=get_left_most_left(node)
    while 1==1
        if node.isleaf~=1
            node=node.children{1};
        else
            left_most_node=node;
            break;
        end
    end
end

function[num]=NumAllNode(node)
    num=length(node.children);
    for i=1:length(node.children)
        num=num+NumAllNode(node.children{i});
    end
end

function[num]=NumMultiChildNode(node)
    num=0;
    if length(node.children)>2
        num=num+length(node.children);
    end
    for i=1:length(node.children)
        num=num+NumMultiChildNode(node.children{i});
    end
end

function[]=PrintNode(node)
    if length(node.children)>2
        for i=1:length(node.children)
            [node.children{i}.nuclear,node.children{i}.relation]
        end
        disp('\n')
    end
    for i=1:length(node.children)
        PrintNode(node.children{i});
    end
end

function[A]=GetNegativeOneNode(node,LeafNodes)
    A=[];
    if node.leaf_index==1
        return
    end
    node1=LeafNodes{node.leaf_index};
    node2=LeafNodes{node.leaf_index-1};
    if node1.parent~=node2.parent
        v1=GetAncestor(node1);
        v2=GetAncestor(node2);
        A=setdiff(v2,v1);
    end
end

function[vector]=GetNegative(node,vector,LeafNodes,AllNodes)
    for i=1:length(LeafNodes)-1
        node1=LeafNodes{i};
        node2=LeafNodes{i+1};
        if node1.parent~=node2.parent
            v1=GetAncestor(node1);
            v2=GetAncestor(node2);
            A1=setdiff(v1,v2);
            A2=setdiff(v2,v1);
            for i=1:length(A1)
                for j=1:length(A2)
                    if AllNodes{A1(i)}.parent~=AllNodes{A2(j)}.parent
                        vector=[vector;[A1(i),A2(j),0]];
                    end
                end
            end
        end
    end
end

function[vector]=GetAncestor(node)
    vector=[node.index];
    while 1==1
        if length(node.parent)~=0
            node=node.parent;
            vector=[vector,node.index];
        else break;
        end
    end
end

function[vector]=GetPositive(node,vector)
    if node.isleaf==1
        return;
    end
    for i=1:length(node.children)
        for j=i+1:length(node.children)
            vector=[vector;[node.children{i}.index,node.children{j}.index,1]];
        end
    end
    for i=1:length(node.children)
        vector=GetPositive(node.children{i},vector);
    end
end

function[AllNodes]=GetAllNode(node,AllNodes)
    node.index=length(AllNodes)+1;
    AllNodes{length(AllNodes)+1}=node;
    for i=1:length(node.children)
        AllNodes=GetAllNode(node.children{i},AllNodes);
    end
end

function[]=Print(node)
    disp('(');
    disp(node.nuclear);
    if(node.relation~=-100)
        disp(' ')
        disp(node.relation);
    end
    if(node.isleaf==1)
        disp(' ')
        disp(node.sen_index);
    end
    for i=1:length(node.children)
        Print(node.children{i});
    end
    disp(')')
end
