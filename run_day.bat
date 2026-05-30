@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ============================================================
:: run_day.bat - FYP1 Automated Day Runner
:: Usage:
::   run_day.bat 1     <- runs Day 1
::   run_day.bat 2     <- runs Day 2
::   run_day.bat all   <- runs all calls
::
:: Call lists are auto-discovered from colab_transcripts/
:: To override, create call_schedule.json in project root
:: ============================================================

set DAY=%1
if "%DAY%"=="" (
    echo Usage: run_day.bat [1^|2^|all]
    exit /b 1
)

call conda activate fyp2
cd /d C:\fyp1_fixed

:: Get call list dynamically from Python
echo Getting call list for Day %DAY%...
python -c "from config import discover_calls; s=discover_calls(); calls=s['day%DAY%'.replace('dayall','all')]; [print(c) for c in (s['all'] if '%DAY%'=='all' else s.get('day%DAY%',[]))]" > .calls_tmp.txt 2>nul

if not exist .calls_tmp.txt (
    echo ERROR: Could not get call list. Check config.py and colab_transcripts\
    exit /b 1
)

:: Count calls
set TOTAL=0
for /f %%c in (.calls_tmp.txt) do set /a TOTAL+=1

if %TOTAL%==0 (
    echo ERROR: No calls found for Day %DAY%
    del .calls_tmp.txt
    exit /b 1
)

:: Pre-flight check
echo.
echo ============================================================
echo  FYP1 - Day %DAY% Runner  ^(%TOTAL% calls^)
echo ============================================================
python preflight_check.py --day %DAY%
if errorlevel 1 (
    echo PREFLIGHT FAILED - fix errors above before running.
    del .calls_tmp.txt
    pause
    exit /b 1
)

:: Backup pipeline_results.json
if exist outputs\latest\pipeline_results.json (
    echo Backing up pipeline_results.json...
    copy /Y outputs\latest\pipeline_results.json outputs\latest\pipeline_results_before_day%DAY%.json > nul
)

:: Run each call
set DONE=0
set FAILED=0

echo.
echo Starting %TOTAL% calls...
echo ============================================================

for /f %%c in (.calls_tmp.txt) do (
    set /a DONE+=1
    echo.
    echo [!DONE!/%TOTAL%] Processing: %%c
    echo ------------------------------------------------------------
    python main.py --skip_transcription --call_id %%c
    if errorlevel 1 (
        echo ERROR: %%c failed
        set /a FAILED+=1
    )
)

del .calls_tmp.txt

:: Save day backup
if "%DAY%"=="all" (
    set BACKUP_NAME=all
) else (
    set BACKUP_NAME=day%DAY%
)
copy /Y outputs\latest\pipeline_results.json outputs\latest\%BACKUP_NAME%.json > nul
echo Backup saved: outputs\latest\%BACKUP_NAME%.json

:: Summary
set /a SUCCESS=%TOTAL%-%FAILED%
echo.
echo ============================================================
echo  Day %DAY% complete: %SUCCESS%/%TOTAL% calls successful
if %FAILED% gtr 0 echo  WARNING: %FAILED% call^(s^) failed
echo ============================================================

:: Run evaluation after Day 2 or all
if "%DAY%"=="2" goto :run_eval
if "%DAY%"=="all" goto :run_eval
goto :end

:run_eval
echo.
echo Running full evaluation pipeline...
python combine_results.py
python compare_labels.py
python evaluate.py
python qa_scorer.py --update_csv
echo.
echo Done! Run: streamlit run dashboard\app.py

:end
echo.
pause
endlocal
