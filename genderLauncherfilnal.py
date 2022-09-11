import sys
from tkinter import Tk
from header_footer import UI

if __name__ == "__main__":
    root = Tk()
    # Create a new client
    app = UI(root)
    root.mainloop()