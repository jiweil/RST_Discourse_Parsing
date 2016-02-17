function[grad]=Backward_Text(tree,grad,parameter,batch,h_edu,lstms,all_c_t,lstms_r,all_c_t_r)
    N=size(batch.Word,1);
    T=size(batch.Word,2);
    h=[];
    for i=1:length(tree.LeafNodes)
        h=[h,tree.LeafNodes{i}.h];
    end
    dh=arrayfun(@tanhPrime,h,grad.leaves);
    grad.compose_W=dh*h_edu';
    dh=parameter.compose_W'*dh;

    d_left_h=dh(1:parameter.dimension,:);
    d_right_h=dh(parameter.dimension+1:parameter.dimension*2,:);
    grad.left_W=zeroMatrix(size(parameter.left_W));
    grad.right_W=zeroMatrix(size(parameter.right_W));
    zeroState=zeroMatrix([parameter.dimension,N]);

    dc=zeroState;
    wordCount = 0;
    if parameter.update_v==1
        numInputWords=size(batch.Word,1)*size(batch.Word,2);
        allEmbGrads=zeroMatrix([parameter.dimension,numInputWords]);
        allEmbGrads_r=zeroMatrix([parameter.dimension,numInputWords]);
    end
    for t=T:-1:1
        unmaskedIds=batch.Left{t};
        if t==1 c_t_1 =zeroState;
        else c_t_1 = all_c_t{t-1};
        end
        c_t = all_c_t{t};
        lstm = lstms{t};
        lstm_grad=lstmUnitGrad(parameter.left_W,lstm, c_t, c_t_1, dc, d_left_h,parameter);
        dc=lstm_grad.dc;
        d_left_h=lstm_grad.input(parameter.dimension+1:end,:);
        if parameter.update_v==1
            embIndices=batch.Word(unmaskedIds,t)';
            embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
            numWords = length(embIndices);
            allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
            allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
            wordCount = wordCount + numWords;
        end
        grad.left_W=grad.left_W+lstm_grad.W;
    end
    if parameter.update_v==1
        allEmbGrads(:, wordCount+1:end) = [];
        allEmbIndices(wordCount+1:end) = [];
        [W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices);
    end


    dc_r=zeroState;
    wordCount=0;
    for t=T:-1:1
        unmaskedIds=batch.Left{t};
        if t==1 c_t_1 =zeroState;
        else c_t_1 = all_c_t_r{t-1};
        end
        c_t = all_c_t_r{t};
        lstm = lstms_r{t};
        lstm_grad=lstmUnitGrad(parameter.right_W,lstm, c_t, c_t_1, dc_r, d_right_h,parameter);
        dc_r=lstm_grad.dc;
        d_right_h=lstm_grad.input(parameter.dimension+1:end,:);
        grad.right_W=grad.right_W+lstm_grad.W;
        if parameter.update_v==1
            embIndices=batch.Word_r(unmaskedIds,t)';
            embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
            numWords = length(embIndices);
            allEmbIndices_r(wordCount+1:wordCount+numWords) = embIndices;
            allEmbGrads_r(:, wordCount+1:wordCount+numWords) = embGrad;
            wordCount = wordCount + numWords;
        end
    end
    if parameter.update_v==1
        allEmbGrads_r(:, wordCount+1:end) = [];
        allEmbIndices_r(wordCount+1:end) = [];
        [W_emb_r, grad.indices_r] = aggregateMatrix(allEmbGrads_r, allEmbIndices_r);
    end

    if parameter.update_v==1
        grad.W_emb=W_emb+W_emb_r;
        grad.W_emb=grad.W_emb/size(tree.clique_vector,1);
    end
    grad.compose_W=grad.compose_W/size(tree.clique_vector,1)+parameter.C*parameter.compose_W;
    grad.left_W=grad.left_W/size(tree.clique_vector,1)+parameter.C*parameter.left_W;
    grad.right_W=grad.right_W/size(tree.clique_vector,1)+parameter.C*parameter.right_W;
end

function[lstm_grad]=lstmUnitGrad(W,lstm, c_t, c_t_1, dc, dh,parameter)
    dc =arrayfun(@plusTanhPrimeTriple,dc,lstm.f_c_t,lstm.o_gate, dh);
    %dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
    do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, lstm.f_c_t, dh);
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);
    df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc);
    lstm_grad.dc = lstm.f_gate.*dc;
    dl = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
    d_ifoa = [di; df; do; dl];
    lstm_grad.W = d_ifoa*lstm.input'; %dw
    lstm_grad.input=W'*d_ifoa;
    if parameter.dropout~=0
        lstm_grad.input=lstm_grad.input.*lstm.drop_left;
    end
    clear dc; clear do; clear di; clear df; clear d_ifoa;
end


