param(
    [Parameter(Mandatory=$true)][ValidateSet('A','D','F','G')][string]$Variant,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$RunArguments
)
$ErrorActionPreference = 'Stop'
$PythonCommand = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { 'python' }
$RunAction = if ($Variant -eq 'A') { 'evaluate' } else { 'run' }
& $PythonCommand (Join-Path $PSScriptRoot 'run_ablation.py') --variant $Variant --action $RunAction @RunArguments
exit $LASTEXITCODE
