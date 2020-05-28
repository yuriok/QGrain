# How to install QGrain

**QGrain** is written by [Python](https://www.python.org/). So, it's **cross-platform**.

This means it can be installed on many kinds of systems, like **Windows**, **Linux**, **Mac OS X**, etc.

Moreover, due to Python's features, there are two ways to install QGrain.

## 1. Use executable file

We have packed the executable setup file for **Windows** users. Because Windows does not have Python internally installed, and Windows users are not used to the command-line interface.

You can download the latest version of QGrain from [here](https://github.com/QGrain-Organization/QGrain/releases).

When the setup file was downloaded, what you need to do is double-clicking the setup file to execute the installation program.

All things are similar to other softwares on **Windows**. And then, you can see the shortcut on the desktop.

## 2. Use Python

This method is for the user who has experience in Python and Shell (i.e. command-line interface). It uses the Python interpreter to execute the source codes of QGrain.

The advantage of this method is its expansibility. Through the `pip` (the package installer for Python), you can easily download and update QGrain. If you are not satisfied with some functionalities of QGrain, or there are some small bugs, you can modify the codes by yourself to solve these problems instead of waiting for a new update to fix them.

1. Install Python

    For Linux and Mac OS X users, you may have the built-in Python3 interpreter. You can run the command `python` or `python3` in your terminal to check if Python is existing.

    Note: Using `python` or `python3` depends on the alias or the filename of your Python3 interpreter, rather than you can choose Python2 or Python3 at liberty. Python2 is too old and has been obsoleted.interpreter

    If you have Python3 installed on your computer, you can see the text like below (test on Ubuntu 18.04 LTS). You can see the version of your Python. And the `>>>` symbols hint you that you have entered the interactive model of Python's interpreter. Type `quit()` to quit this model and back to the terminal.

    ```bash
    your_user_name:~$ python3
    Python 3.6.9 (default, Nov  7 2019, 10:44:02)
    [GCC 8.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    ```

    If you have not Python3 yet, you can visit the [official website of Python](https://www.python.org/downloads/) to download and install it.

    Note: If you are a **Windows** user, remember to check the **Add Python to Windows PATH** option while installing Python. If not, some commands in this tutorial may be **invalid**, due to the `PATH` mechanism of Windows.

2. Get source codes and install QGrain

    We have upload QGrain to [PyPI](https://pypi.org/). So, you can get the codes very expediently through `pip`. By running the following command, you can get QGrain module installed.

    `pip install QGrain`

    For Chinese users, this command below may be faster.

    `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple QGrain`

    Note: On some systems, the name of `pip` of Python3 may be `pip3`.

    Run `pip install -U QGrain` to check and update QGrain.

    In addition, you can download the codes from Github. The link of our repository is over [here](https://github.com/QGrain-Organization/QGrain).
      * You can clone the repository by running the following command if you have `git` installed.

          `git clone https://github.com/QGrain-Organization/QGrain.git`

      * Or, you can download the pure codes directly by clicking [here](https://github.com/QGrain-Organization/QGrain/archive/master.zip).

          If you choose to download the zip file, please extract it.

    Then, you should change the directory to QGrain's folder, and run the following command.

    `python setup.py install` or `pip install .`

3. Run QGrain

    Now, everything is ok.

    Then you can run `qgrain` command in your terminal.

    If it goes well, you can see the app interface below.

    If the command `qgrain` was not found. It may be caused by the lack of Python's directory in the Windows `PATH` environment variable.

    Because QGrain will generate an executable file (e.g. qgrain.exe on Windows) to start this app, it's located at the `Scripts` sub-folder of your Python. If this folder is not in your `PATH`, the system can not find this executable file, and the command will raise the not found error.

    In order to let the modification of `PATH` come into force, you need to restart your PC.

    If restart did not solve your problem, you may need to check the `PATH` variable and **APPEND** the root and `Scripts` folders of Python to the `PATH`.

    For some users who do not want to modify the `PATH`, there also is another way to start QGrain.

    1. Type `python` or `python3` to enter your Python interpreter.

    2. Type following codes.

        ```python
        import QGrain
        QGrain.main()
        ```

    ![App Appearance](../figures/app_appearance.png)
