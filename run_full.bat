@echo off
REM =============================================================
REM  Run full evaluation pipeline: 7 modes x 3 datasets
REM  Usage: run_full.bat [model] [workers]
REM  Example: run_full.bat gpt-4o 15
REM =============================================================

set MODEL=%1
set WORKERS=%2
if "%MODEL%"=="" set MODEL=gpt-4o
if "%WORKERS%"=="" set WORKERS=10

set UV=C:\Users\Administrator\.local\bin\uv.exe

echo ===========================================
echo  MODEL: %MODEL% ^| WORKERS: %WORKERS%
echo  RUNNING: BASELINE + ENHANCED + S1-S6
echo ===========================================

echo.
echo [1/8] BASELINE
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS%
if errorlevel 1 echo WARNING: baseline had errors

echo.
echo [2/8] ENHANCED
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --enhanced
if errorlevel 1 echo WARNING: enhanced had errors

echo.
echo [3/8] S1: Prompt Rewrite
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s1_prompt
if errorlevel 1 echo WARNING: s1_prompt had errors

echo.
echo [4/8] S2: Tool Compression
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s2_compress
if errorlevel 1 echo WARNING: s2_compress had errors

echo.
echo [5/8] S3: Chain-of-Thought
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s3_cot
if errorlevel 1 echo WARNING: s3_cot had errors

echo.
echo [6/8] S4: Two-Stage LLM
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s4_twostage
if errorlevel 1 echo WARNING: s4_twostage had errors

echo.
echo [7/8] S5: Few-Shot
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s5_fewshot
if errorlevel 1 echo WARNING: s5_fewshot had errors

echo.
echo [8/8] S6: Context-Aware Pipeline
echo -------------------------------------------
%UV% run python main.py --dataset all --model %MODEL% --workers %WORKERS% --strategy s6_context
if errorlevel 1 echo WARNING: s6_context had errors

echo.
echo ===========================================
echo  All modes done. Rescoring...
echo ===========================================
%UV% run python rescore.py

echo.
echo ===========================================
echo  Generating RESULTS.md...
echo ===========================================
%UV% run python generate_results_md.py

echo.
echo ===========================================
echo  ALL DONE! Check results/ and RESULTS.md
echo ===========================================
pause
