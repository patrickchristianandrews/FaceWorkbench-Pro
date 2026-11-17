import socket,threading,webbrowser
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
def main():
    port=free_port(); config=uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'); server=uvicorn.Server(config)
    threading.Thread(target=server.run,daemon=True).start()
    import webview
    webview.create_window('FaceWorkbench Pro',f'http://127.0.0.1:{port}/',width=1360,height=860,min_size=(1024,680),js_api=DesktopBridge())
    webview.start()
if __name__=='__main__': main()
