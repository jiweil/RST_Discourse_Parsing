function[Batches]=ReadText(filename)
    fd=fopen(filename);
    Batches={};
    tline = fgets(fd);
    i=1;
    batch_index=0;
    Texts={};
    while ischar(tline)
        if length(deblank(tline))==0
            batch_index=batch_index+1;
            Batches{batch_index}=deal_text(Texts);
            i=1;
            Texts={};
        else
            Texts{i}=str2num(deblank(tline));
            i=i+1;
        end
        tline = fgets(fd);
    end
end

function[batch]=deal_text(Texts)
    batch.MaxLenSource=0;
    for i=1:length(Texts)
        if length(Texts{i})>batch.MaxLenSource
            batch.MaxLenSource=length(Texts{i});
        end
    end
    batch.Word=ones(length(Texts),batch.MaxLenSource);
    batch.Word_r=ones(length(Texts),batch.MaxLenSource);
    Delete=zeros(length(Texts),batch.MaxLenSource);
    for j=1:length(Texts)
        source_length=length(Texts{j});
        batch.Word(j,batch.MaxLenSource-source_length+1:batch.MaxLenSource)=Texts{j};
        batch.Word_r(j,batch.MaxLenSource-source_length+1:batch.MaxLenSource)=wrev(Texts{j});
        Delete(j,1:batch.MaxLenSource-source_length)=1;
    end
    for j=1:batch.MaxLenSource
        batch.Delete{j}=find(Delete(:,j)==1);
        batch.Left{j}=find(Delete(:,j)==0);
    end
end
