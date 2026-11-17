"""Run on Windows to produce dist/FaceWorkbench/FaceWorkbench.exe. PyInstaller does not cross-compile."""
import os,sys
from pathlib import Path
from PyInstaller.__main__ import run
if __name__=='__main__':
    args=['desktop.py','--name=FaceWorkbench','--windowed','--onedir','--noconfirm',f'--add-data=static{os.pathsep}static',
          '--collect-all=webview','--hidden-import=uvicorn.logging','--hidden-import=uvicorn.loops.auto',
          '--hidden-import=uvicorn.protocols.http.auto','--hidden-import=uvicorn.lifespan.on']
    try: import onnxruntime; args+=['--collect-all=onnxruntime']
    except ImportError: pass
    try: import faiss; args+=['--collect-all=faiss']
    except ImportError: pass
    args+=['--hidden-import=sklearn.linear_model']
    run(args)
