function[h]=lstmUnit_tree(parameter,node,h_left,h_right,isTraining)%lstm unit calculation
    h_left=node.children{1}.h;
    h_right=node.children{2}.h;
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
    c_t=f_l_gate.*node.children{1}.c+f_r_gate.*node.children{2}.c+i_gate.*a_signal;
    f_c_t = parameter.nonlinear_f(c_t);
    h=o_gate.*f_c_t;
    node.h=h;
    
    node.c=c_t;
    node.lstm.input = input;
    node.lstm.i_gate = i_gate;
    node.lstm.f_l_gate = f_l_gate;
    node.lstm.f_r_gate = f_r_gate;
    node.lstm.o_gate=o_gate;
    node.lstm.a_signal = a_signal;
    node.lstm.f_c_t = f_c_t;
    if isTraining==1&&parameter.dropout~=0
        node.lstm.drop_left=drop_left;
    end
end

