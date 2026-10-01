' Run the QuantLab dashboard (read-only, localhost only: http://127.0.0.1:8765/live) with NO window
' and WAIT for it. Launched by the Windows Task Scheduler task "QuantLab dashboard" (at logon).
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
rc = CreateObject("WScript.Shell").Run("cmd.exe /c """ & here & "\dashboard.cmd""", 0, True)
WScript.Quit rc
