import os,sys
from typing import Union,Tuple
import json

from RenNet import RyLog,RyCore


# PLOT Core
from RenNet.RyCore import GPU,CPU
from RenNet.RyCore import getTimestr
from RenNet.RyCore import getcfp,loadConfig,Runner
from RenNet.RyLog import check



def getDefaultRunSetting(s:"__file__")->Tuple['Runner','RyCore.getDebugOut','DBLogger']:
    """
    runner,logger,dblogger
    """
    try:
        config = loadConfig(s)
    except FileNotFoundError as e:
        print(e)
        config = {}
    runner = Runner(filestr=s)

    if "log_path" not in config:
        log_path = os.path.join(runner.base_path,runner.base_name+'.log')
    else:
        log_path = config["log_path"]
    logger = getLogger(log_path)

    if "db_path" not in config:
        db_path = os.path.join(runner.base_path,runner.base_name+'.db')
    else:
        db_path =config["db_path"]

    dblogger = DBLogger(db_path)
    return runner,logger,dblogger

# PLOT Save
class DBLogger(RyLog.DB):
    base_table_name = 'OUTPUT'
    base_columns = ["runindex","timestr", "tag","data"]
    base_cols_sql = "uid integer PRIMARY KEY AUTOINCREMENT, runindex int,timestr text, tag text, data json"
    uid_name = 'uid'
    

    def __init__(self,db_path):
        super().__init__({"db_path":db_path})
        
        
        q = self.query
        order = "SELECT MAX(runindex) from {}".format(self.base_table_name)
        check(q.prepare,order)
        check(q.exec_)
        q.next()
        runindex = q.value(0)
        if runindex =='':
            runindex = 0
        self._current_runindex = runindex +1

    def isValidTag(self,tag:str):
        return tag in ('url,psnr,loss,config')
    
    def indexOfColumn(self,column):
        assert column in self.base_columns
        return self.base_columns.index(column)
    
    def forward(self,i=1):
        """runindex+=i
        """
        self._current_runindex +=i
    
    @property
    def current_runindex(self): # from 0
        return self._current_runindex


    def send(self,values = None,*,timestr = None,tag = None,data:dict = None):
            
        runindex = self.current_runindex

        if values is not None:
            i = self.indexOfColumn("runindex")
            values = list(values).insert(i,runindex)

            j = self.indexOfColumn("tag")
            assert self.isValidTag(values[j])

            k = self.indexOfColumn("data")
            ss = json.dumps(values[k])
            values[k] = ss

            return self._send(values)
        else:

            assert self.isValidTag(tag)
            data = json.dumps(data)
            self._send(None,{
                "runindex":runindex,
                "timestr":timestr,
                "tag":tag,
                "data":data
            })

# PLOT .log
def getLogger(out_path):
    fmt = "%(asctime)s %(levelname)s %(filename)s %(lineno)d %(process)d :: %(message)s"
    logger = RyCore.getDebugOut(out_path,mode = 'cf',fmt = fmt,datefmt = None)

    return logger
        