function[all_h,lstms,all_c_t,lstms_r,all_c_t_r]=Forward_Text(batch,parameter,isTraining)
    N=size(batch.Word,1);
    T=batch.MaxLenSource;
    zeroState=zeroMatrix([parameter.dimension,N]);
    for t=1:T
        all_c_t{t}=zeroState;
        all_c_t_r{t}=zeroState;
    end
    for t=1:T
        if t==1
            h_t_1=zeroState;
            c_t_1 =zeroState;
        else 
            h_t_1=h;
            c_t_1=all_c_t{t-1};
        end
        x_t=parameter.vect(:,batch.Word(:,t));
        x_t(:,batch.Delete{t})=0;
        h_t_1(:,batch.Delete{t})=0;
        c_t_1(:,batch.Delete{t})=0;
        [lstms{t},h,all_c_t{t}]=lstmUnit_sequence(parameter.left_W,parameter,x_t,h_t_1,c_t_1,isTraining);
    end
    for t=1:T
        if t==1
            h_t_1=zeroState;
            c_t_1 =zeroState;
        else 
            h_t_1=h_r;
            c_t_1=all_c_t_r{t-1};
        end
        x_t=parameter.vect(:,batch.Word_r(:,t));
        x_t(:,batch.Delete{t})=0;
        h_t_1(:,batch.Delete{t})=0;
        c_t_1(:,batch.Delete{t})=0;
        [lstms_r{t},h_r,all_c_t_r{t}]=lstmUnit_sequence(parameter.right_W,parameter,x_t,h_t_1,c_t_1,isTraining);
    end
    all_h{1}=parameter.nonlinear_f(parameter.compose_W*[h;h_r]);
    all_h{2}=[h;h_r];
end


