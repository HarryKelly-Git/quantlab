' Start the QuantLab PAPER runner supervisor (scripts\run_paper.cmd) with NO window, so there is
' nothing on the taskbar that can be closed by accident (closing the console window killed the bot
' on 2026-10-01). Used by the "QuantLab paper runner" shortcut in the user Startup folder.
' Check it:  the dashboard (scripts\dashboard.cmd -> http://127.0.0.1:8765/live)
' Stop it:   .venv\Scripts\python -m quantlab.cli paper stop   (the supervisor then exits)
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
CreateObject("WScript.Shell").Run "cmd.exe /c """ & here & "\run_paper.cmd""", 0, False
