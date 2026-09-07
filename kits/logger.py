import sys
import os


class DoubleWriter(object):
    def __init__(self, log_file, mode="w"):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass

        if not self.log.closed:
            try:
                self.log.write(message)
                self.log.flush()
            except ValueError:
                pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass

        if not self.log.closed:
            try:
                self.log.flush()
            except ValueError:
                pass

    def close(self):
        if not self.log.closed:
            self.log.close()


def setup_logger(save_dir, args):
    log_file = getattr(args, "log_file", None)
    if log_file is None:
        log_file = os.path.join(save_dir, "logs", args.model, args.dataset_name, f"{args.run_id}.txt")

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = DoubleWriter(log_file, mode="w")

    print(f"[Log] file={log_file}")
    print()

    return log_file
