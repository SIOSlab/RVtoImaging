import logging


# ANSI escape sequences for colors
class ColorCodes:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


# Custom formatter to add colors
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: ColorCodes.BLUE,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.MAGENTA,
    }

    def format(self, record):
        log = super().format(record)
        color = self.COLORS.get(record.levelno, ColorCodes.WHITE)
        # See https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for
        # info on the color codes
        return f"\033[38;5;193m\033[48;5;16m[RVtoImaging]\033[0m {color}{log}"


logger = logging.getLogger(__name__)

shell_handler = logging.StreamHandler()
file_handler = logging.FileHandler("debug.log")

logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

shell_fmt = "%(levelname)s [%(asctime)s] \033[0m%(message)s"
file_fmt = (
    "[RVtoImaging] %(levelname)s %(asctime)s [%(filename)s:"
    "%(funcName)s:%(lineno)d] %(message)s"
)
shell_formatter = ColorFormatter(shell_fmt)
file_formatter = logging.Formatter(file_fmt)

shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)

logger.propagate = False
