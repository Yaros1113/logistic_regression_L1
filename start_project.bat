@echo off
echo Installation has started...
echo.

:: Проверяем, существует ли уже папка env
if exist "env\" (
    echo Virtual environment 'env' already exists.
    echo Activating existing virtual environment...
    call env\Scripts\activate.bat
    echo Existing virtual environment activated!
    echo.
    goto :skip_installation
)

echo Virtual environment not found. Checking for Python 3.13.*...

:: Пробуем разные варианты вызова Python
set "python_cmd=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    set "python_cmd=py"
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Python is not installed or added to PATH
        echo Install Python 3.13.* and add it to your PATH environment variable
        pause
        exit /b 1
    )
)

:: Получаем версию Python
for /f "tokens=2" %%i in ('%python_cmd% --version 2^>^&1') do set "python_version=%%i"

:: Проверяем, что версия начинается с 3.13
echo %python_version% | findstr /r "^3\.13\." >nul
if errorlevel 1 (
    echo Error: Found Python version %python_version%, but Python 3.13.* is required.
    echo Install Python 3.13.*
    pause
    exit /b 1
)

echo Python 3.13.* found. Creating virtual environment...
echo.

echo Creating a virtual environment...
%python_cmd% -m venv env

:skip_installation

echo Activating virtual environment...
call env\Scripts\activate.bat

echo Virtual environment created and activated!
echo.

echo Updating pip to the latest version...
python -m pip install --upgrade pip

if %errorlevel% neq 0 (
    echo Warning: Failed to update pip. Continuing with current version...
    pip --version
) else (
    echo Pip updated successfully!
    pip --version
)
echo.

echo Installing dependencies from requirements.txt...
:: Проверяем существование requirements.txt
if not exist "requirements.txt" (
    echo Error: requirements.txt not found!
    echo Please make sure requirements.txt exists in the current directory.
    pause
    exit /b 1
)

:: Устанавливаем зависимости
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies!
    echo Check your requirements.txt file and internet connection.
    pause
    exit /b 1
)

echo Dependencies installed successfully!
echo.

echo Backend installation completed!
echo Running application...
echo.

python src/main.py

pause