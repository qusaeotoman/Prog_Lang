<#
build.ps1 — helper for assembling + downloading + executing via Portable.RemoteTasks.Manager.exe

- Defaults = как в твоих командах
- Все параметры можно переопределять
- Дожидается сборки: -g ... -r ... -o ... ретраит каждые 1 сек
- TaskId берёт из БУФЕРА КОНСОЛИ (WinAPI), без редиректов stdout/stderr => не ломает "Неверный дескриптор"
#>

[CmdletBinding()]
param(
  # Path to Portable.RemoteTasks.Manager.exe (relative to ProjectDir or absolute)
  [string]$ManagerExe = ".\Portable.RemoteTasks.Manager.exe",

  # Credentials
  [string]$UserLogin    = "475770",
  [string]$UserPassword = "b6521087-b349-48b8-a2db-30bfe53889bd",

  # Working directory
  [string]$ProjectDir = (Get-Location).Path,

  # Assemble
  [string]$AssembleService = "AssembleDebug",
  [string]$DefinitionFile  = "stack16.target.pdsl",
  [string]$ArchName        = "stack16",
  [string]$AsmListing      = "test.asm",
  [string]$SourcesDir      = "C:\Users\User\Downloads\parsing_grammar\architecture\",

  # Download result
  [string]$RemoteBinary = "out.ptptb",
  [string]$LocalBinary  = "out_local.ptptb",
  [int]$PollDelayMs     = 1000,
  [int]$MaxPollAttempts = 0,   # 0 = infinite

  # Execute
  [string]$ExecuteService     = "ExecuteBinaryWithInput",
  [switch]$WaitForExecute     = $true,  # adds -w
  [string]$BinaryFileToRun    = "out_local.ptptb",
  [string]$IpRegStorageName   = "ip",
  [string]$FinishMnemonicName = "hlt",
  [string]$CodeRamBankName    = "code",
  [string]$StdinRegStName     = "rin",
  [string]$StdoutRegStName    = "rout",
  [string]$InputFile          = "in.txt",

  # Like "> file"
  [string]$OutputFile = "file",

  # Extra args to allow "all parameters"
  [string[]]$AssembleExtraArgs = @(),
  [string[]]$GetExtraArgs      = @(),
  [string[]]$ExecuteExtraArgs  = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function NowStamp() { (Get-Date).ToString("yyyy-MM-dd HH:mm:ss") }

function Resolve-PathMaybeRelative([string]$p, [string]$baseDir) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $p }
  if ([IO.Path]::IsPathRooted($p)) { return $p }
  return (Join-Path -Path $baseDir -ChildPath $p)
}

# Try to enlarge console buffer so the printed GUID doesn't scroll away
try {
  $raw = $Host.UI.RawUI
  $bs  = $raw.BufferSize
  if ($bs.Height -lt 5000) {
    $raw.BufferSize = New-Object Management.Automation.Host.Size($bs.Width, 5000)
  }
} catch {
  # ignore
}

# --- WinAPI console buffer reader ---
if (-not ("ConsoleBufferReader" -as [type])) {
  Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class ConsoleBufferReader {
  [StructLayout(LayoutKind.Sequential)]
  public struct COORD { public short X; public short Y; }

  [StructLayout(LayoutKind.Sequential)]
  public struct SMALL_RECT { public short Left; public short Top; public short Right; public short Bottom; }

  [StructLayout(LayoutKind.Sequential)]
  public struct CONSOLE_SCREEN_BUFFER_INFO {
    public COORD dwSize;
    public COORD dwCursorPosition;
    public short wAttributes;
    public SMALL_RECT srWindow;
    public COORD dwMaximumWindowSize;
  }

  const int STD_OUTPUT_HANDLE = -11;

  [DllImport("kernel32.dll", SetLastError=true)]
  static extern IntPtr GetStdHandle(int nStdHandle);

  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool GetConsoleScreenBufferInfo(IntPtr hConsoleOutput, out CONSOLE_SCREEN_BUFFER_INFO lpConsoleScreenBufferInfo);

  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  static extern bool ReadConsoleOutputCharacterW(
    IntPtr hConsoleOutput,
    StringBuilder lpCharacter,
    uint nLength,
    COORD dwReadCoord,
    out uint lpNumberOfCharsRead
  );

  public static CONSOLE_SCREEN_BUFFER_INFO GetInfo() {
    var h = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(h, out info)) {
      throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
    }
    return info;
  }

  public static string[] ReadLines(int startY, int endY) {
    var info = GetInfo();
    int width = info.dwSize.X;
    int maxY = info.dwSize.Y - 1;

    if (startY < 0) startY = 0;
    if (endY > maxY) endY = maxY;
    if (endY < startY) return new string[0];

    var h = GetStdHandle(STD_OUTPUT_HANDLE);
    int count = endY - startY + 1;
    var lines = new string[count];

    for (int i = 0; i < count; i++) {
      int y = startY + i;
      var sb = new StringBuilder(width);
      uint read;
      var coord = new COORD { X = 0, Y = (short)y };
      ReadConsoleOutputCharacterW(h, sb, (uint)width, coord, out read);
      lines[i] = sb.ToString().TrimEnd('\0').TrimEnd();
    }
    return lines;
  }
}
"@
}

function Invoke-ManagerConsoleSegment {
  param(
    [Parameter(Mandatory=$true)] [string]$ExePath,
    [Parameter(Mandatory=$true)] [string[]]$ArgumentList,
    [int]$FallbackTailLines = 250
  )

  # Snapshot cursor before
  $pre = $null
  try { $pre = [ConsoleBufferReader]::GetInfo() } catch { $pre = $null }

  # Run without any stdout redirection/pipes (keeps real console handles)
  $p = Start-Process -FilePath $ExePath -ArgumentList $ArgumentList -WorkingDirectory $ProjectDir -NoNewWindow -Wait -PassThru
  $exitCode = $p.ExitCode

  # Snapshot cursor after
  $post = $null
  try { $post = [ConsoleBufferReader]::GetInfo() } catch { $post = $null }

  $text = ""

  if ($pre -and $post) {
    $startY = [int]$pre.dwCursorPosition.Y
    $endY   = [int]$post.dwCursorPosition.Y

    # If scrolled/wrapped, fall back to tail
    if ($endY -lt $startY) {
      $startY = [Math]::Max(0, $endY - $FallbackTailLines)
    } else {
      # include a couple lines above for safety
      $startY = [Math]::Max(0, $startY - 2)
    }

    $lines = [ConsoleBufferReader]::ReadLines($startY, $endY)
    $text  = ($lines -join "`n").Trim()
  }
  else {
    # Cannot read console (wrong host), return empty text
    $text = ""
  }

  return [pscustomobject]@{
    ExitCode = $exitCode
    Text     = $text
  }
}

function Extract-LastGuid([string]$text) {
  if ([string]::IsNullOrWhiteSpace($text)) { return $null }

  $m = [regex]::Matches($text, "(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
  if ($m.Count -eq 0) { return $null }
  return $m[$m.Count - 1].Value
}

# --- MAIN ---
$ProjectDir = Resolve-PathMaybeRelative $ProjectDir (Get-Location).Path
Set-Location -LiteralPath $ProjectDir

$exePath = Resolve-PathMaybeRelative $ManagerExe $ProjectDir
if (-not (Test-Path -LiteralPath $exePath)) {
  throw "ManagerExe not found: $exePath"
}

Write-Host "[$(NowStamp)] ProjectDir: $ProjectDir"
Write-Host "[$(NowStamp)] SourcesDir: $SourcesDir"

# 1) Assemble
$assembleArgs = @(
  "-ul", $UserLogin,
  "-up", $UserPassword,
  "-s",  $AssembleService,
  "definitionFile", $DefinitionFile,
  "archName",       $ArchName,
  "asmListing",     $AsmListing,
  "sourcesDir",     $SourcesDir
) + $AssembleExtraArgs

Write-Host "[$(NowStamp)] START Assemble ($AssembleService)..."
$assembleRes = Invoke-ManagerConsoleSegment -ExePath $exePath -ArgumentList $assembleArgs

# Read GUID from console segment (this is the key fix)
$taskId = Extract-LastGuid $assembleRes.Text
if (-not $taskId) {
  Write-Host "----- captured segment (from console buffer) -----"
  Write-Host $assembleRes.Text
  Write-Host "-------------------------------------------------"
  throw "Cannot find TaskId GUID in Assemble console output. Make sure you run in normal PowerShell/Windows Terminal (not ISE)."
}

Write-Host "[$(NowStamp)] TaskId: $taskId"

# 2) Poll download result (-g ... -r ... -o ...)
$attempt = 0
$localOutPath = Join-Path -Path $ProjectDir -ChildPath $LocalBinary

Write-Host "[$(NowStamp)] WAIT result: -g $taskId -r $RemoteBinary -o $LocalBinary"
while ($true) {
  $attempt++

  $getArgs = @(
    "-ul", $UserLogin,
    "-up", $UserPassword,
    "-g",  $taskId,
    "-r",  $RemoteBinary,
    "-o",  $LocalBinary
  ) + $GetExtraArgs

  $getRes = Invoke-ManagerConsoleSegment -ExePath $exePath -ArgumentList $getArgs

  $fileOk = (Test-Path -LiteralPath $localOutPath) -and ((Get-Item -LiteralPath $localOutPath).Length -gt 0)

  if ($getRes.ExitCode -eq 0 -and $fileOk) {
    Write-Host "[$(NowStamp)] OK: result downloaded -> $localOutPath"
    break
  }

  Write-Host "[$(NowStamp)] NOT READY / FAIL (attempt $attempt). ExitCode=$($getRes.ExitCode)"
  if ($getRes.Text) {
    Write-Host "----- manager said -----"
    Write-Host $getRes.Text
    Write-Host "------------------------"
  }

  if ($MaxPollAttempts -gt 0 -and $attempt -ge $MaxPollAttempts) {
    throw "MaxPollAttempts reached ($MaxPollAttempts)."
  }

  Start-Sleep -Milliseconds $PollDelayMs
}

# 3) Execute (stdout -> file)  — как у тебя: "> file"
$execArgs = @(
  "-ul", $UserLogin,
  "-up", $UserPassword
)
if ($WaitForExecute) { $execArgs += "-w" }

$execArgs += @(
  "-s", $ExecuteService,
  "definitionFile",     $DefinitionFile,
  "archName",           $ArchName,
  "binaryFileToRun",    $BinaryFileToRun,
  "ipRegStorageName",   $IpRegStorageName,
  "finishMnemonicName", $FinishMnemonicName,
  "codeRamBankName",    $CodeRamBankName,
  "stdinRegStName",     $StdinRegStName,
  "stdoutRegStName",    $StdoutRegStName,
  "inputFile",          $InputFile
) + $ExecuteExtraArgs

$outPath = Join-Path -Path $ProjectDir -ChildPath $OutputFile
Write-Host "[$(NowStamp)] START Execute ($ExecuteService)... stdout -> $outPath"

# Here redirection is expected/needed (like your command)
& $exePath @execArgs 1> $outPath
$execCode = $LASTEXITCODE

if ($execCode -ne 0) {
  Write-Host "[$(NowStamp)] EXECUTE FAILED. ExitCode=$execCode"
  exit $execCode
}

Write-Host "[$(NowStamp)] DONE. ExitCode=0"
exit 0
