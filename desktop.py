import socket,threading,webbrowser,sys,time,urllib.request
from pathlib import Path
import uvicorn
from registry import ROOT
from app import app

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); return s.getsockname()[1]
class DesktopBridge:
    def save_file_dialog(self,filename='export.json'):
        import webview
        w=webview.windows[0]; result=w.create_file_dialog(webview.SAVE_DIALOG,save_filename=filename)
        return result[0] if result else None
    def select_folder_dialog(self):
        import webview
        r=webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        return r[0] if r else None
    def open_external_url(self,url):
        if url.startswith('https://'): webbrowser.open(url)
def create_server(port):
    # Frozen --windowed executables have no sys.stdout/sys.stderr.
    # Uvicorn's default logging formatter assumes stderr.isatty() exists.
    config=uvicorn.Config(app,host='127.0.0.1',port=port,
                          log_level='warning',log_config=None,access_log=False)
    return uvicorn.Server(config)

def wait_for_server(port, timeout=20):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status',timeout=1.0) as response:
                if response.status==200:return True
        except (OSError,TimeoutError):pass
        time.sleep(0.15)
    return False

def main():
    port=free_port()
    server=create_server(port)
    threading.Thread(target=server.run,daemon=True).start()
    if not wait_for_server(port):
        raise RuntimeError('FaceWorkbench local API did not start')
    if '--self-test' in sys.argv:
        from registry import MODELS,MODEL_REGISTRY
        import cv2
        yunet=MODELS/MODEL_REGISTRY['YuNet']
        sface=MODELS/MODEL_REGISTRY['SFace']
        if not yunet.is_file() or not sface.is_file():
            raise RuntimeError('Required bundled YuNet/SFace models are missing')
        detector=cv2.FaceDetectorYN.create(str(yunet),'',(320,320))
        recognizer=cv2.FaceRecognizerSF.create(str(sface),'')
        if detector is None or recognizer is None:
            raise RuntimeError('Bundled ONNX models could not be loaded by OpenCV')
        server.should_exit=True
        return
    import webview
    webview.create_window('FaceWorkbench Pro',f'http://127.0.0.1:{port}/',width=1360,height=860,min_size=(1024,680),js_api=DesktopBridge())
    webview.start()
if __name__=='__main__': main()
