function[parameter]=ReadRelation(parameter)
    fd=fopen('../data/relation_match');
    tline = fgets(fd);
    parameter.IndexToRelation=[];
    parameter.RelationToIndex=zeros(19,19);
    index=1;
    while ischar(tline)
        M=str2num(deblank(tline));
        parameter.IndexToRelation=[parameter.IndexToRelation;M];
        parameter.RelationToIndex(M(1),M(2))=index;
        index=index+1;
        tline = fgets(fd);
    end
end
