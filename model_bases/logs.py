import mlogger
from sacred import Experiment
from sacred.observers import SqlObserver
from tensorboardX import SummaryWriter

DB_FILE = 'sqlite:///experiment_logs/sacred.db'
exp = Experiment()
exp.observers.append(SqlObserver(DB_FILE))
writer = SummaryWriter(logdir='experiment_logs/tensorboard_runs/')

mlog = mlogger.Container(sacred_exp=exp)
mlog.epoch = mlogger.metric.Simple()

mlog.train = mlogger.Container()
mlog.train.timer = mlogger.metric.Timer()
mlog.train.error = mlogger.metric.Average(summary_writer=writer, plot_title='Train Error')

mlog.val = mlogger.Container()
mlog.val.timer = mlogger.metric.Timer()

mlog.test = mlogger.Container() 