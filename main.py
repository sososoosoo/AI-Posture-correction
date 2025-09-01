# =============================== main.py ===============================
# GUI 실행 진입점
# - 버튼으로 "정면 자세 (Mediapipe)" / "측면 자세 (OpenVINO YOLO)" 중 하나를 선택
# - 선택된 스크립트만 별도 프로세스로 실행해서 메모리/의존성 충돌 방지
# - 가상환경(.venv)이 아닐 때 자동으로 .venv 파이썬으로 재실행(선택사항 구현)

import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_FRONT = PROJECT_ROOT / "final_front.py"
SCRIPT_SIDE  = PROJECT_ROOT / "final_side.py"

def in_venv() -> bool:
    # venv 감지 (Windows/일반 파이썬 호환)
    return hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)

def relaunch_with_venv_if_needed():
    """
    메인 스크립트가 .venv가 아닌 전역 파이썬으로 실행되면,
    .venv\Scripts\python.exe 로 자신을 다시 실행. (사용자 더블클릭 실행 대비)
    """
    if in_venv():
        return  # 이미 venv
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        # 현재 인자 그대로 전달
        args = [str(venv_python), str(PROJECT_ROOT / "main.py"), *sys.argv[1:]]
        try:
            subprocess.call(args, cwd=str(PROJECT_ROOT))
            sys.exit(0)
        except Exception as e:
            messagebox.showerror("재실행 실패", f".venv 파이썬으로 재실행 실패:\n{e}\n"
                                 f"터미널에서 아래로 실행해주세요.\n{venv_python} main.py")
            # 계속 진행(전역 파이썬으로 실행)
    # .venv가 없다면 그냥 진행

class LauncherGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("자세 어때")
        self.root.geometry("420x220")
        self.root.resizable(False, False)

        # 상태
        self.proc = None
        self.running_mode = None

        # UI
        title = tk.Label(root, text="자세 교정 프로그램", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(16, 8))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)

        self.btn_front = tk.Button(
            btn_frame, text="정면 자세 (Mediapipe)", width=22, height=2,
            command=self.launch_front
        )
        self.btn_front.grid(row=0, column=0, padx=8)

        self.btn_side = tk.Button(
            btn_frame, text="측면 자세 (OpenVINO YOLO)", width=26, height=2,
            command=self.launch_side
        )
        self.btn_side.grid(row=0, column=1, padx=8)

        self.lbl_status = tk.Label(root, text="상태: 대기 중", fg="#333333", font=("Segoe UI", 10))
        self.lbl_status.pack(pady=8)

        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=6)

        self.btn_stop = tk.Button(
            ctrl_frame, text="실행 중지", width=10, command=self.stop_running, state=tk.DISABLED
        )
        self.btn_stop.grid(row=0, column=0, padx=6)

        self.btn_exit = tk.Button(
            ctrl_frame, text="프로그램 종료", width=10, command=self.on_exit
        )
        self.btn_exit.grid(row=0, column=1, padx=6)

        # 창 가운데 배치
        self.center_window()

        # 닫기 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 3)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def set_controls_running(self, running: bool):
        if running:
            self.btn_front.config(state=tk.DISABLED)
            self.btn_side.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
        else:
            self.btn_front.config(state=tk.NORMAL)
            self.btn_side.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def launch(self, mode: str, script: Path):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("알림", "이미 실행 중입니다. 먼저 '실행 중지'를 눌러 종료하세요.")
            return
        if not script.exists():
            messagebox.showerror("오류", f"스크립트를 찾을 수 없습니다:\n{script}")
            return

        python_exe = sys.executable  # 현재 인터프리터 (venv 포함)
        try:
            self.proc = subprocess.Popen(
                [python_exe, str(script)],
                cwd=str(PROJECT_ROOT)
            )
            self.running_mode = mode
            self.lbl_status.config(text=f"상태: {mode} 실행 중 (PID {self.proc.pid})")
            self.set_controls_running(True)
            # 주기적으로 상태 폴링
            self.root.after(600, self.poll_process)
        except Exception as e:
            self.proc = None
            self.running_mode = None
            messagebox.showerror("실행 실패", f"{mode} 실행에 실패했습니다:\n{e}")

    def launch_front(self):
        self.launch("정면", SCRIPT_FRONT)

    def launch_side(self):
        self.launch("측면", SCRIPT_SIDE)

    def poll_process(self):
        if self.proc is None:
            return
        code = self.proc.poll()
        if code is None:
            # 아직 실행 중
            self.root.after(600, self.poll_process)
        else:
            # 종료됨
            msg = f"{self.running_mode} 모드가 종료되었습니다. (코드 {code})"
            self.lbl_status.config(text=f"상태: {msg}")
            self.proc = None
            self.running_mode = None
            self.set_controls_running(False)

    def stop_running(self):
        if self.proc is None or self.proc.poll() is not None:
            self.lbl_status.config(text="상태: 대기 중")
            self.set_controls_running(False)
            return
        try:
            self.proc.terminate()
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.lbl_status.config(text="상태: 종료 요청됨…")

    def on_exit(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("종료 확인", "실행 중인 모드를 종료하고 프로그램을 닫을까요?"):
                return
            self.stop_running()
        self.root.after(200, self.root.destroy)

def main():
    # 가상환경 자동 재실행 시도 (더블클릭 실행 보호)
    relaunch_with_venv_if_needed()

    root = tk.Tk()
    app = LauncherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()