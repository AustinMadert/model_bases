import mlogger

mlog = mlogger.Container()
mlog.epoch = mlogger.metric.Simple()

mlog.train = mlogger.Container()
mlog.train.timer = mlogger.metric.Timer()
mlog.train.error = mlogger.metric.Average()

mlog.val = mlogger.Container()
mlog.val.timer = mlogger.metric.Timer()

mlog.test = mlogger.Container()