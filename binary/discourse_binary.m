function[]=discourse_binary()
    addpath('../misc');
    parameter.alpha=0.01;
    parameter.gpu_index=1;
    gpuDevice(parameter.gpu_index);
    parameter.dimension=300;
    parameter.dropout=0.25;
    parameter.CheckGrad=0;
    if parameter.CheckGrad==1&&parameter.dropout~=0
        parameter.drop_left=randSimpleMatrix([2*parameter.dimension,1])<1-parameter.dropout;
    end
    parameter.Initial=0.1;
    parameter.C=0;
    parameter.update_v=0;
    parameter.nonlinear_gate_f = @sigmoid;
    parameter.nonlinear_gate_f_prime = @sigmoidPrime;
    parameter.nonlinear_f = @tanh;
    parameter.nonlinear_f_prime = @tanhPrime;
    Train_Tree='../data/train_tree.txt';
    Train_Text='../data/train_text.txt';
    Train_Trees=ReadTrees(Train_Tree,1);
    Train_Text=ReadText(Train_Text);
    parameter
    [parameter,ada]=Initial(parameter);
    length(Train_Trees)
    length(Train_Text)
    for iter=1:4
        iter
        tic
        for i=1:length(Train_Trees)
            tree=Train_Trees{i};
            batch=Train_Text{i};
            [h_edu,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,1);
            Forward_Tree(tree.root,parameter,h_edu{1},1);
            [grad,cost]=softmax(tree,parameter);
            grad=Backward_Tree(tree,grad,parameter);
            grad=Backward_Text(tree,grad,parameter,batch,h_edu{2},lstms,all_c_t,lstms_r,all_c_t_r);
            tree.root=Free(tree.root);
            clear h_edu; clear lstms; clear all_c_t; clear lstms_r; clear all_c_t_r;
            if parameter.CheckGrad==1
                check(grad,tree,batch,parameter);
            end
            [parameter,ada]=update(parameter,ada,grad);
        end
        file_name=strcat('save1/',num2str(parameter.dropout),'_',int2str(iter),'.mat');
        save('-v7.3',file_name,'parameter');
        toc
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

function[parameter,ada]=update(parameter,ada,grad)
    ada.left_W=ada.left_W+grad.left_W.^2;
    ada.right_W=ada.right_W+grad.right_W.^2;
    ada.compose_W=ada.compose_W+grad.compose_W.^2;
    ada.tree_W=ada.tree_W+grad.tree_W.^2;
    ada.U=ada.U+grad.U.^2;
    ada.W=ada.W+grad.W.^2;
    if parameter.update_v==1
        ada.vect(:,grad.indices_r)=ada.vect(:,grad.indices_r)+grad.W_emb.^2;
        parameter.vect(:,grad.indices_r)=parameter.vect(:,grad.indices_r)-parameter.alpha*arrayfun(@divide_square,grad.W_emb,ada.vect(:,grad.indices_r));
    end
    
    parameter.left_W=parameter.left_W-parameter.alpha*arrayfun(@divide_square,grad.left_W,ada.left_W);
    parameter.right_W=parameter.right_W-parameter.alpha*arrayfun(@divide_square,grad.right_W,ada.right_W);
    parameter.compose_W=parameter.compose_W-parameter.alpha*arrayfun(@divide_square,grad.compose_W,ada.compose_W);
    parameter.tree_W=parameter.tree_W-parameter.alpha*arrayfun(@divide_square,grad.tree_W,ada.tree_W);
    parameter.U=parameter.U-parameter.alpha*arrayfun(@divide_square,grad.U,ada.U);
    parameter.W=parameter.W-parameter.alpha*arrayfun(@divide_square,grad.W,ada.W);

end

function[parameter,ada]=Initial(parameter)
    small=1e-40;
    parameter.left_W=randomMatrix(parameter.Initial,[4*parameter.dimension,2*parameter.dimension]);
    ada.left_W=small*oneMatrix(size(parameter.left_W));
    parameter.right_W=randomMatrix(parameter.Initial,[4*parameter.dimension,2*parameter.dimension]);
    ada.right_W=small*oneMatrix(size(parameter.right_W));
    parameter.compose_W=randomMatrix(parameter.Initial,[parameter.dimension,2*parameter.dimension]);
    ada.compose_W=small*oneMatrix(size(parameter.compose_W));
    
    parameter.tree_W=randomMatrix(parameter.Initial,[5*parameter.dimension,2*parameter.dimension]);
    ada.tree_W=small*oneMatrix(size(parameter.tree_W));

    parameter.W=randomMatrix(parameter.Initial,[parameter.dimension,2*parameter.dimension]);
    ada.W=small*oneMatrix(size(parameter.W));
    parameter.U=randomMatrix(parameter.Initial,[1,parameter.dimension]);
    ada.U=small*oneMatrix(size(parameter.U));

    parameter.vect=gpuArray(load('../data/discourse_wordvector.txt'))';
    %parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,17031]);
    ada.vect=small*oneMatrix(size(parameter.vect));
end
