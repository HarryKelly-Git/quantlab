' Run the QuantLab PAPER runner supervisor (scripts\run_paper.cmd) with NO window and WAIT for it,
' passing its exit code back. Launched by the Windows Task Scheduler task "QuantLab paper runner"
' (at logon, and on demand), so the bot is never a child of an interactive or tool shell that might
' end and take it down. Exit 0 = clean stop (quantlab paper stop); anything else = Task Scheduler
' restarts the task after 1 minute.
' Check it: the dashboard (http://127.0.0.1:8765/live). Stop it: .venv\Scripts\python -m quantlab.cli paper stop
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
rc = CreateObject("WScript.Shell").Run("cmd.exe /c """ & here & "\run_paper.cmd""", 0, True)
WScript.Quit rc
