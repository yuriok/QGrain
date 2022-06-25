from QGrain.ui.AboutDialog import AboutDialog
from QGrain.ui.MainWindow import setup_app

if __name__ == "__main__":
    import sys

    app, splash = setup_app()

    main = AboutDialog()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())
