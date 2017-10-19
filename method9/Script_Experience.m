clc;
clear;

again=true;
while(again)
    try 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%  放入實驗的內容  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%         example 

        yourfunction=@QSMOTE2_1;
%         Performance_Wilcoxon( yourfunction,"yourFunctionName","outTextFileName")
        Performance_Wilcoxon( yourfunction,"QSMOTE","outTextFileName")
        
%         yourfunction=@QSMOTE2_2;
%         Performance_Wilcoxon( yourfunction,"yourFunctionName2","outTextFileName2")

%         yourfunction=@QSMOTE2_1;
%         Performance_t_test( yourfunction,"yourFunctionName","outTextFileName")

%         yourfunction=@QSMOTE2_2;
%         Performance_t_test( yourfunction,"yourFunctionName2","outTextFileName2")
    

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        again=false;
    catch err
        disp(['errorMessage : ' err.message]);
        disp('sleep 60 sec, if you dont want to wait , please click Ctrl+c  ');
        disp(" ");
        pause(60);
        again=true;
    end    
end