
xm2(){
    rsync -avz -Ph -e 'ssh -p 10220' --exclude='*/global_step*/*' -L /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data root@10.0.193.220:/data1/aimodels/v1004-bak/dbgpt-hub/
}


rsync -avz -Ph --update -e 'ssh -p 10220' --exclude='。git' -L root@62.234.207.173:/root/space/repos/eosphoros-ai/DB-GPT-Hub/ /Users/johnsaxon/test/github.com/oushu1zhangxiangxuan1/eosphoros-ai/DB-GPT-Hub/ 

v~2BtH.X8Cmgj