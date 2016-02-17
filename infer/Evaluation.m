function[]=Evaluation()
    addpath('../misc');
    addpath('../binary');
    gpu_index=1;
    gpuDevice(gpu_index);
    Test_Text='../data/test_text.txt';

    IndexToRelation=ReadRelation();
    Test_Text=ReadText(Test_Text);
    load '../data/TestSpan.mat';
    load '../data/TestRelation.mat';
    binary=load('../binary/save1/0.25_4.mat','parameter');
    binary=binary.parameter;
    multi=load('../multi/save1/0.25_4.mat','parameter');
    multi=multi.parameter;
    Infer(Test_Text,binary,multi,TestRelation,TestSpan,IndexToRelation);
end

function[IndexToRelation]=ReadRelation()
    fd=fopen('../data/relation_match');
    tline = fgets(fd);
    IndexToRelation=[];
    while ischar(tline)
        M=str2num(deblank(tline));
        IndexToRelation=[IndexToRelation;M];
        tline = fgets(fd);
    end
end
