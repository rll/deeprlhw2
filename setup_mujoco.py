#!/usr/bin/env python
import os, os.path as osp, subprocess, sys, shutil

os.chdir("vendor")

def cap(cmd):
    "call and print"
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.call(cmd,shell=True)


if sys.platform.startswith("darwin"):
    platname = "osx"
elif sys.platform.startswith("linux"):
    platname = "linux"
elif sys.platform.startswith("windows"):
    platname = "win"
fname = "mjpro120DEEPRL_%s.zip"%platname

assert osp.exists(fname),"Please download %s and place in vendor directory"%fname

targdir = "mujoco_%s"%platname
if osp.exists(targdir):
    shutil.rmtree(targdir)


cap("unzip %s"%(fname))
shutil.move("mjpro120DEEPRL", targdir)
if sys.platform.startswith("darwin"):
    shutil.rmtree("__MACOSX")

shutil.copy("LICENSE_DEEPRL.txt",targdir)