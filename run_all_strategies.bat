@echo off
REM ===================================================
REM  Usage: run_all_strategies.bat <model_name> [limit] [workers]
REM  Example: run_all_strategies.bat qwen 100 10
REM           run_all_strategies.bat gemma 50 5
REM ===================================================

set MODEL=%1
set LIMIT=%2
set WORKERS=%3

if "%MODEL%"=="" (
    echo ERROR: Please specify a model name.
    echo Usage: run_all_strategies.bat ^<model_name^> [limit] [workers]
    echo Available models: qwen, gemma, glm, lfm, nemotron, llama
    pause
    exit /b 1
)
if "%LIMIT%"=="" set LIMIT=100
if "%WORKERS%"=="" set WORKERS=10

echo ===================================================
echo  MODEL: %MODEL%  LIMIT: %LIMIT%  WORKERS: %WORKERS%
echo  RUNNING: BASELINE + ENHANCED + 5 STRATEGIES
echo ===================================================

echo.
echo [1/7] BASELINE
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS%

echo.
echo [2/7] ENHANCED
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --enhanced

echo.
echo [3/7] STRATEGY 1: Prompt Rewrite
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --strategy s1_prompt

echo.
echo [4/7] STRATEGY 2: Tool Compression
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --strategy s2_compress

echo.
echo [5/7] STRATEGY 3: Chain-of-Thought
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --strategy s3_cot

echo.
echo [6/7] STRATEGY 4: Two-Stage LLM
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --strategy s4_twostage

echo.
echo [7/7] STRATEGY 5: Few-Shot
python main.py --dataset all --model %MODEL% --limit %LIMIT% --workers %WORKERS% --strategy s5_fewshot

echo.
echo ===================================================
echo  DONE! Model=%MODEL% Results in results/
echo ===================================================

echo.
echo Rescoring results...
python rescore.py

pause
