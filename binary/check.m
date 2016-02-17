function check(grad,tree,batch,parameter)
    if 1==0
    disp('check_U')
    check_U(1,grad.U(1),tree,batch,parameter);
    check_U(150,grad.U(150),tree,batch,parameter);
    end
    if 1==0
    disp('check_W')
    check_W(1,1,grad.W(1,1),tree,batch,parameter);
    check_W(1,301,grad.W(1,301),tree,batch,parameter);
    end
    if 1==0
    disp('check_tree_W')
    check_tree_W(1,1,grad.tree_W(1,1),tree,batch,parameter);
    check_tree_W(1+parameter.dimension,1,grad.tree_W(1+parameter.dimension,1),tree,batch,parameter);
    check_tree_W(1+parameter.dimension*2,1,grad.tree_W(1+parameter.dimension*2,1),tree,batch,parameter);
    check_tree_W(1+parameter.dimension*3,1,grad.tree_W(1+parameter.dimension*3,1),tree,batch,parameter);
    check_tree_W(1+parameter.dimension*4,1,grad.tree_W(1+parameter.dimension*4,1),tree,batch,parameter);
    end
    if 1==0
    disp('check_compose_W')
    check_compose_W(1,1,grad.compose_W(1,1),tree,batch,parameter);
    check_compose_W(1,1+parameter.dimension,grad.compose_W(1,1+parameter.dimension),tree,batch,parameter);
    end
    if 1==0
    disp('check_left_W')
    check_left_W(1,1,grad.left_W(1,1),tree,batch,parameter);
    check_left_W(1+parameter.dimension,1,grad.left_W(1+parameter.dimension,1),tree,batch,parameter);
    check_left_W(1+2*parameter.dimension,1,grad.left_W(1+2*parameter.dimension,1),tree,batch,parameter);
    check_left_W(1+3*parameter.dimension,1,grad.left_W(1+3*parameter.dimension,1),tree,batch,parameter);
    end
    if 1==0
    disp('check_right_W')
    check_right_W(1,1,grad.right_W(1,1),tree,batch,parameter);
    check_right_W(1+parameter.dimension,1,grad.right_W(1+parameter.dimension,1),tree,batch,parameter);
    check_right_W(1+2*parameter.dimension,1,grad.right_W(1+2*parameter.dimension,1),tree,batch,parameter);
    check_right_W(1+3*parameter.dimension,1,grad.right_W(1+3*parameter.dimension,1),tree,batch,parameter);
    end
    if 1==0
    disp('vheck_v')
    for i=1:size(grad.W_emb,2)
        check_v(1,grad.indices_r(i),grad.W_emb(1,i),tree,batch,parameter);
    end
    end
end

function check_v(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value=(cost1-cost2)/(2*e);
    [value,grad_value]
    value-grad_value
end

function check_right_W(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.right_W(i,j)=parameter.right_W(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.right_W(i,j)=parameter.right_W(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.right_W(i,j)=parameter.right_W(i,j)+e;
    value=(cost1-cost2)/(2*e);
    value;
    grad_value;
    value-grad_value
end

function check_left_W(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.left_W(i,j)=parameter.left_W(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.left_W(i,j)=parameter.left_W(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.left_W(i,j)=parameter.left_W(i,j)+e;
    value=(cost1-cost2)/(2*e);
    value;
    grad_value;
    value-grad_value
end

function check_tree_W(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.tree_W(i,j)=parameter.tree_W(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.tree_W(i,j)=parameter.tree_W(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.tree_W(i,j)=parameter.tree_W(i,j)+e;
    value=(cost1-cost2)/(2*e);
    value
    grad_value
    value-grad_value
end

function check_U(i,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.U(i)=parameter.U(i)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.U(i)=parameter.U(i)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.U(i)=parameter.U(i)+e;
    value=(cost1-cost2)/(2*e);
    value;
    grad_value;
    value-grad_value
end

function check_compose_W(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.compose_W(i,j)=parameter.compose_W(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.compose_W(i,j)=parameter.compose_W(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.compose_W(i,j)=parameter.compose_W(i,j)+e;
    value=(cost1-cost2)/(2*e);
    value;
    grad_value;
    value-grad_value
end

function check_W(i,j,grad_value,tree,batch,parameter)
    e=0.0001;
    parameter.W(i,j)=parameter.W(i,j)+e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost1]=softmax(tree,parameter);
    parameter.W(i,j)=parameter.W(i,j)-2*e;
    [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
    Forward_Tree(tree.root,parameter,h_edu{1},1);
    [grad,cost2]=softmax(tree,parameter);
    parameter.W(i,j)=parameter.W(i,j)+e;
    value=(cost1-cost2)/(2*e);
    value;
    grad_value;
    value-grad_value
end

