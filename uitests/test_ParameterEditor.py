from QGrain.ui.ParameterEditor import ParameterEditor
from QGrain.ui.MainWindow import setup_app

if __name__ == "__main__":
    import sys

    app, splash = setup_app()

    main = ParameterEditor()
    main.show()
    splash.finish(main)
    import numpy as np
    print(np.array(main.parameters).shape)

    sys.exit(app.exec())
