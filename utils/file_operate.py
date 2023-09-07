import os
import shutil

class FileOperate:

    def mkdirs(self, dir, is_rm=True):
        try:
            if os.path.exists(dir):
                if is_rm:
                    self.del_dir(dir)
                    os.makedirs(dir)
            else:
                os.makedirs(dir)
        except:
            return

    def is_type(self, path, types=[]):

        file = os.path.basename(path)
        name, extension = os.path.splitext(file)
        if extension not in types:
            return False
        return True

    def del_dir(self, path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except:
                return