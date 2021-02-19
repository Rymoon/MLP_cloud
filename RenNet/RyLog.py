'''
2021-2-5
Reforge version `RenLog`.
'''

from datetime import date,datetime
import time
import os,sys
import json
from typing import Union
from enum import Enum
from warnings import warn
from random import random
from pprint import pprint
import itertools

import PySide2
from PySide2.QtSql import QSql
from PySide2.QtSql import QSqlDatabase, QSqlQuery,QSqlDriver
from PySide2.QtCore import Qt,QAbstractTableModel,QModelIndex

import pygit2
from pygit2 import discover_repository,Repository,Signature
from pygit2 import GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED
from pygit2.credentials import KeypairFromAgent

from RenNet.RyCore import getTimestr
from RenNet.RyCore import getcfp,loadConfig,Runner
from RenNet.RyCore import getDebugOut

# PLOT config
config = loadConfig(__file__)

# PLOT Logger


class LoggerError(Exception):
    def __init__(self,msg):
        msg = "{}".format(msg)
        super().__init__(msg)

def check(func,*args):
    if not func(*args):
        raise ValueError(func.__self__.lastError())

def addSqlPlaceholder(order):
    '''
    "sss{}sss"->"sss?sss"
    '''
    assert "{{}}" not in order
    order = order.replace('{}','{0}')
    return order.format('?')

def jsonDumpNotNone(v):
    v= json.dumps(v) if v is not None else None
    return v

# PLOT Base class
class DB:
    base_table_name = ''
    base_columns = []
    uid_name = ''
    extra_table_name = []
    
    def __init__(self,config:dict = config):
        """
        config["db_path"]
        """
        self.db_path = None
        self.query = None
        self.config = config
        self.open()

    @staticmethod
    def placeholder(n:int):
        return ','.join('?'*n)

    def createTable(self,table_name,columns_sql):
        db = self.db
        if table_name in db.tables():
            warn('Tabel {} already exists.'.format(table_name))
            return 
        order = "CREATE TABLE {0} ({1})".format(table_name,columns_sql)
        q = self.query
        check(q.exec_,order)
    
    def open(self):
        '''Open database, create empty one if not exist.

        Need self.config  .
        '''
        self.db_path = self.config['db_path']
        db = QSqlDatabase.addDatabase("QSQLITE",self.db_path)
        db.setDatabaseName(self.db_path)
        check(db.open)
        self.query = QSqlQuery(db)

        db.transaction()
        try:
            if self.base_table_name not in db.tables():
                self.createTable(self.base_table_name,self.base_cols_sql)

        except Exception as e:
            db.rollback()
            raise e
        else:
            db.commit()
    @property
    def db(self):
        db = QSqlDatabase.database(self.db_path)
        return db

    def getQuery(self):
        q = QSqlQuery(self.db)
        return q

    def _insert(self,table_name,columns,values):
        q = self.query
        SQL_INSERT = """
INSERT INTO {} ({}) VALUES({})
""" 
        n_col = len(values)

        cols = columns
        order = SQL_INSERT.format(table_name,','.join(cols),self.placeholder(n_col))
        #print(order)
        #print(values)

        check(q.prepare,order)
        
        for v in values:
            q.addBindValue(v)
        #print(q.boundValues())
        check(q.exec_)
        return q.lastInsertId()
    
    def _send(self,values,karg_values):
        q = self.query
        table_name = self.base_table_name
        columns = self.base_columns

        if values is None:
            if type(karg_values) is list:
                values = karg_values
            elif type(karg_values) is dict:
                values = [karg_values[k] for k in self.base_columns]
        else:
            assert len(values)==len(columns)
        id = self._insert(table_name,columns,values)
        return id

    def _find(self,table_name,column,value):
        q = self.query
        order = "select * from {} where {}={}".format(table_name,column,'?')
        check(q.prepare,order)
        q.addBindValue(value)
        check(q.exec_)

        result = []

        while q.next():
            result.append(q.value(self.uid_name))
        
        return result

    def find(self,column,value):
        return self._find(self.base_table_name,column,value)


class MainHandler(DB):
    pass


class SubHandler:
    table_name = ''
    columns = []
    extra_columns_sql = ""
    def __init__(self,main_handler:MainHandler):
        self.main_handler = main_handler
        self.open()

    def open(self):
        db= self.db
        if self.table_name not in db.tables():
            db.transaction()
            try:
                extra_template = "uid integer,{{}},FOREIGN KEY(uid) REFERENCES {}(uid)".format(self.main_handler.base_table_name)

                self.createTable(self.table_name,extra_template.format(self.extra_columns_sql))
            except Exception as e:
                db.rollback()
                raise e
            else:
                db.commit()

    def createTable(self,table_name,columns_sql):
        self.main_handler.createTable(table_name,columns_sql)

    def _send(self,values,base_values,extra_values):
        q = self.query

        if values is None:
            pass
        else:
            n_col = len(self.columns)
            assert len(values)==n_col+len(self.main_handler.base_columns)

            base_values = values[:-n_col]
            extra_values = values[-n_col:]

        id = self.main_handler.send(base_values)
        self.main_handler._insert(self.table_name,["uid"]+list(self.columns),[id]+list(extra_values))

        return id

    def find(self,column,value):
        return self.main_handler._find(self.table_name,column,value)

    @property
    def query(self):
        return self.main_handler.query

    @property
    def db(self):
        return self.main_handler.db



# PLOT App Class
class Logger(MainHandler):
    base_table_name= 'MAIN_LOG'
    base_columns = ["timestr","filefrom" ,"runstr", "msg" , "storagelevel"]
    base_cols_sql =  "uid integer PRIMARY KEY AUTOINCREMENT, timestr text, filefrom text,runstr text, msg text, storagelevel integer"
    uid_name = 'uid'  # 手动使用，改这个没用
    extra_table_name = []

    def __init__(self,config:dict = config):
        super().__init__(config) 

    
    def send(self,values=None,*,timestr=None,filefrom=None,runstr=None,msg=None,storagelevel=None):

        return self._send(values,[timestr,filefrom,runstr,msg,storagelevel])


class TrainSettingHandler(SubHandler):
    table_name = 'TRAIN_SETTING'
    columns = ["valuedict"]
    extra_columns_sql = "valuedict text"

    def __init__(self,logger:Logger):
        super().__init__(logger)
        

    def send(self,values=None,*,timestr=None,filefrom=None,runstr=None,msg=None,storagelevel=None,valuedict:dict=None):
        if values is not None:
            values = list(values)
            values[-1] = jsonDumpNotNone(values[-1])

        valuedict = jsonDumpNotNone(valuedict)
        return self._send(values,[timestr,filefrom,runstr,msg,storagelevel],[valuedict])

class TrainLogHandler(SubHandler):
    table_name = 'TRAIN_LOG'
    columns = ["commit_oid","train_setting_uid"]
    extra_columns_sql = "commit_oid text,train_setting_uid integer ,FOREIGN KEY(train_setting_uid) REFERENCES TRAIN_SETTING(uid)"

    def __init__(self,logger:Logger):
        super().__init__(logger)



    def send(self,values=None,*,timestr=None,filefrom=None,runstr=None,msg=None,storagelevel=None,commit_oid=None,train_setting_uid=None):
        return self._send(values,[timestr,filefrom,runstr,msg,storagelevel],[commit_oid,train_setting_uid])

class TrainStateHandler(SubHandler):
    table_name = "TRAIN_STATE"
    columns = ["state","progress","report","train_log_uid"]
    extra_columns_sql = "state text,progress json,report json,train_log_uid,  FOREIGN KEY(train_log_uid) REFERENCES TRAIN_LOG(uid)"

    def __init__(self,logger:Logger):
        super().__init__(logger)


    def send(self,values=None,*,timestr=None,filefrom=None,runstr=None,msg=None,storagelevel=None,state=None,progress=None,report:dict=None,train_log_uid=None):
        if values is not None:
            values = list(values)
            values[-2] = jsonDumpNotNone(values[-2])
        report = jsonDumpNotNone(report)
        return self._send(values,[timestr,filefrom,runstr,msg,storagelevel],[state,progress,report,train_log_uid])
        

# PLOT Git

def getRenGitCheck(git_folders :dict= None):
    if git_folders is None:
        import ADMMNet
        import RenNetData
        git_folders={
        'RenNet':GitInfo(getcfp(__file__)),
        'ADMMNet':GitInfo(getcfp(ADMMNet.__file__)),
        'RenNetData':GitInfo(getcfp(RenNetData.__file__)),
    }

  
    oid_info = {}
    dirty_info = {}
    for k,g in git_folders.items():
        g.collect()
        dirty_info[k]= g.dirty
        oid_info[k] =g.current_oid_hex

    return oid_info,dirty_info



def commitAll(repo,msg,is_first=False):
    index = repo.index
    status = repo.status()
    for filepath, flag in status.items():
        if flag == GIT_STATUS_WT_DELETED:
            msg = 'Remove file %s' % filepath
            del index[filepath]
            
        elif flag == GIT_STATUS_WT_NEW:
            msg = 'Add file %s' % filepath
            index.add(filepath)
            
        elif flag == GIT_STATUS_WT_MODIFIED:
            msg = 'Change file %s' % filepath
            index.add(filepath)
       # else:
            #raise Exception([filepath,statusFlag(flag)])
    cmt = doCommit(repo, index, msg,is_first)
    return cmt

def doCommit(repo, index, msg,is_first=False):
    index.write()
    tree = index.write_tree()
    committer = Signature('gitsync', 'root@localhost')
    ref = 'HEAD' 
    parents = [] if is_first else [repo.head.target]
    sha = repo.create_commit(
        ref, committer, committer, msg, tree, parents)
    commit = repo[sha]
    return commit

s = 'GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED'
names = s.split(',')
from pygit2 import GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED

values = [GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED]

def statusFlag(flag_int):
    s = 'GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED'
    names = s.split(',')

    values = [GIT_STATUS_CURRENT,GIT_STATUS_INDEX_NEW,GIT_STATUS_INDEX_MODIFIED,GIT_STATUS_INDEX_DELETED,GIT_STATUS_INDEX_RENAMED,GIT_STATUS_INDEX_TYPECHANGE,GIT_STATUS_WT_NEW,GIT_STATUS_WT_MODIFIED,GIT_STATUS_WT_DELETED,GIT_STATUS_WT_TYPECHANGE,GIT_STATUS_WT_RENAMED,GIT_STATUS_WT_UNREADABLE,GIT_STATUS_IGNORED,GIT_STATUS_CONFLICTED]

    iii = None
    result = []
    for i,v in enumerate(values):
        if v&flag_int!=0:
            result.append(names[i])
    return result



class GitInfo:
    '''
    Call .current_oid_hex to save commit-oid as str.
    '''
    def __init__(self,base_path=None):
        self.base_path = None
        self.repo = None
        self.dirty = None
        self.ignored = None
        self.status = None
        self.diff = None
        self.head_name = None
        self.current_commit = None
        self.current_oid_hex = None

        if base_path is not None:
            self.open(base_path)

    def open(self,base_path):
        self.base_path= base_path
        repo_path = discover_repository(self.base_path)
        self.repo = Repository(repo_path)

        try:
            self.repo.head.target
        except Exception as e:
            raise Exception('repo.head not found! No commit in repo.')

    def collect(self):
        '''Compare worktree to `HEAD`.
            Show difference, and HEAD commit info.
            No branches check.
        '''
        if self.repo is None:
            repo_path = discover_repository(self.base_path)
            self.repo = Repository(repo_path)
        
        # diff

        # [patch,patch,...]
        # patch.text :str
        self.diff= list(self.repo.diff('HEAD')) 


        # status

        self.status = self.repo.status()
        ss = self.status
        stt_p = []
        stt_p_ig = []
        stt_f = []
        for filepath, flag in ss.items():
            flag = statusFlag(flag)
            if 'GIT_STATUS_IGNORED' in flag:
                stt_p_ig.append(filepath)
            else:
                stt_p.append(filepath)
                stt_f.append(flag)
        
        self.dirty = [(p,f) for p,f in zip(stt_p,stt_f)]
        self.ignored = stt_p_ig

        # branch,ref

        self.head_name = self.repo.head.name

        # commit

        # commit attrs:
        #     author,commiter: Signature:  .name .email
        #     commit_time # Unixtimestamp
        self.current_commit = self.repo.head.peel()
        self.current_oid_hex = self.current_commit.oid.hex

    def report(self):
        self.collect()
        data = {}

        #diff = [patch.text for patch in self.diff] 
        #data['diff']= diff

        status = {'dirty':self.dirty,
        'ignored':self.ignored}

        data['status'] = status

        ref = self.head_name
        data['head'] = ref

        c = self.current_commit
        t = c.commit_time
        utc_str = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

        commit = {
            'message':c.message,
            'oid':c.oid,
            'time':utc_str,
            'author':(c.author.name,c.author.email),
            'committer':(c.committer.name,c.author.email),
        }

        data['commit']= commit

        return data

    def commit(self,msg):
        return commitAll(self.repo,msg,is_first=False)

    def __str__(self):
        return str(self.report())



if __name__=='__main__':
    runner = Runner(getcfp(__file__))
    
    dbpath = os.path.join(runner.base_path,"RyLog_test.db")

    myconfig = {"db_path":dbpath}
    lg = Logger(myconfig)
    setting = TrainSettingHandler(lg)
    train = TrainLogHandler(lg)
    state = TrainStateHandler(lg)

    # create train setting
    timestr = getTimestr()
    TO_STORE = 1
    runstr = 'module test'

    v = [timestr,__file__,runstr,'Create Train Setting',TO_STORE,{'args':[1,2,3]}]
    train_setting_uid = setting.send(v)

    # create train task

    git_info = GitInfo(runner.base_path)
    git_info.collect()
    pprint('git_info')
    pprint(git_info.report(),width=80)

    commit_oid = git_info.current_commit.oid
    N = 5

    v = [timestr,__file__,runstr,'Create Train Task',TO_STORE,commit_oid,train_setting_uid]
    train_log_uid =train.send(v)

    # start train task

    v = [timestr,__file__,runstr,'Start Train Task',TO_STORE, 'START','0/{}'.format(N),'Everthing is OK',train_log_uid]
    state.send(v)

    for i in range(N):
        time.sleep(0.1)

        timestr = getTimestr()

        mid_report = {'psnr':random()*2+28}
        v = [timestr,__file__,runstr,'Running Train Task',TO_STORE, 'Running','{}/{}'.format(i,N),mid_report,train_log_uid]
        state.send(v)

    v = [timestr,__file__,runstr,'Finish Train Task',TO_STORE, 'FINISH','{0}/{0}'.format(N),'Everthing is OK',train_log_uid]
    state.send(v)

    # after train task finished

    save_report = [{'path':'result path here'}]
    v = timestr,__file__,runstr,'Result Saved.',TO_STORE, 'SAVE','1/1',save_report,train_log_uid
    state.send(v)
    
    # create view for plot
    q = lg.query
    db = lg.db
    view_name = "psnr_data"

    doSql= lambda order,*args: check(q.exec_,order.format(*args))
    if view_name in db.tables(QSql.Views):
        doSql("DROP VIEW {}",view_name)
    doSql("""   CREATE VIEW {} AS
                Select uid,report,train_log_uid,json_extract(report,'$.psnr') as psnr
                From TRAIN_STATE AS T
                WHERE json_valid(T.report)==1 AND psnr is not NULL""",view_name)
    doSql("Select Distinct train_log_uid from {}",view_name)
    train_log_uid_list = []
    while q.next():
        train_log_uid_list.append(q.value(0))
    train_psnr = {}
    for i in train_log_uid_list:
        doSql("Select psnr From {} Where cast (train_log_uid as integer) == {}",view_name,i)
        train_psnr[i] = []
        while q.next():
            train_psnr[i].append(q.value(0))


    
    ppp = getDebugOut(os.path.join(runner.base_path,"RyLog_test.md"),mode=  'cf',).send

    ppp('train_uid | psnr report')
    ppp('-----|------')
    for k,v in train_psnr.items():
        ppp('{}|{}'.format(k,v))
    ppp('> You can copy these to MarkDown Browser.')

