"""
Interactive programm creating a file.lst in the PATH folder (or reading other .lst files if any)
and interactively puts lines given from the user to this file.
You can add and delete lines or save changes on every step.
"""

import os

class CancelledError(Exception): pass

PATH = "/home/ksu/"

def set_path():
    os.chdir(PATH)
    print("You are in ", PATH)

def get_listd(li):
    listd = {}
    for i, item in enumerate(sorted(li), start=1):
        listd[i] = item
    return listd

def print_listd(li):
    listd = get_listd(li)
    for key, value in listd.items():
        print("{}: {}".format(key, value))

def get_filelist_dict(PATH):
    res = []
    files = os.listdir(PATH)
    for file in files:
        if file.endswith(".lst"):
            res.append(file)
    if res:
        print_listd(res)
        return get_listd(res)

def get_integer(msg, default=None, minimum=0, maximum=1000):
    msg += ": " if not default else "[{}]: ".format(default)
    while True:
        num = input(msg)
        if not num:
            if default is not None:
                return default
            else:
                raise CancelledError
        try:
            num = int(num)
        except ValueError:
            print("Enter a number.")
            continue
        if minimum <= num <= maximum:
            return num
        else:
            print("The number should be less than {}"
                  "and greater than {}.".format(maximum, minimum))
            continue

def get_string(msg, default=None):
    msg += ": " if not default else " [{}]: ".format(default)
    line = input(msg)
    if not line:
        if default:
            return default
        else:
            raise CancelledError
    return line

def create_new_file():
    new_file_msg = "Enter a new filename"
    filename = get_string(new_file_msg)
    if not filename.endswith(".lst"):
        filename += ".lst"
    with open(filename, "w") as file:
        file.write("")

def is_file_empty(line):
    return True if not len(line) else False

def choose_file(filedict):
    filenum_msg = "Choose filename number or 0 to create a new file"
    while True:
        num = get_integer(filenum_msg, minimum=0, maximum=len(filedict))
        if num is 0:
            create_new_file()
            continue
        elif 0 < num <= len(filedict):
            filepath = PATH + filedict[num]
            read_file(filepath)
            return 0

def read_file(filepath):
    saved = False
    while True:
        with open(filepath, "r+") as file:
            lines = []
            for line in file:
                lines.append(line.strip())
            if is_file_empty(lines):
                print("-- no items are in the list --")
            ask_action(file, filepath, lines, saved)

def ask_action(file, filepath, lines, saved):
    while True:
        print()
        if is_file_empty(lines):
            possible_actions = "aq"
            ask_msg = "[A]dd [Q]uit"
        elif saved:
            possible_actions = "adq"
            ask_msg = "[A]dd [D]elete [Q]uit"
        else:
            possible_actions = "adsq"
            ask_msg = "[A]dd [D]elete [S]ave [Q]uit"

        print_listd(lines)
        action = get_string(ask_msg, default="a")
        action = action.lower()
        if not is_possible_action(action, possible_actions):
            continue
        (lines, saved) = read_action(file, filepath, lines, action, saved)

def save_changes(file, filepath, lines):
    save_msg = "Save unsaved changes (y/n)"
    to_save = get_string(save_msg, "y")
    if to_save.lower() in {"y", "yes"}:
        read_action(file, filepath, lines, "s", False)

def is_possible_action(action, possible_actions):
    if action not in possible_actions:
        pos_string = ""
        for ch in possible_actions:
            pos_string += ch
            pos_string += ch.upper()
        print("ERROR: invalid choice--enter one of '{}'".format(pos_string))
        input("Press Enter to continue...")
        print()
        return False
    return True

def read_action(file, filepath, lines, action, saved):
    if action == "q":
        if not saved:
            save_changes(file, filepath, lines)
        raise End
    elif action == "a":
        lines = add_line(lines)
        saved = False
    elif action == "d":
        lines = delete_line(lines)
        saved = False
    elif action == "s":
        file.close()
        with open(filepath, "w") as file:
            for line in lines:
                file.write(line)
        saved = True
        print("The file {} is saved.".format(filepath))
    return (lines, saved)

def add_line(lines):
    msg = "Add item"
    line = get_string(msg)
    lines.append(line)
    return lines

def delete_line(lines):
    dict = get_listd(lines)
    delete_msg = "Delete item number (or 0 to cancel)"
    is_delete = get_integer(delete_msg, minimum=0, maximum=len(dict))
    if is_delete:
        delete_string = dict.pop(is_delete)
        lines.remove(delete_string)
    return lines

def main():
    set_path()
    while True:
        try:
            filedict = get_filelist_dict(PATH)
            flist_len = len(filedict)
            if not flist_len:
                continue
            choose_file(filedict)
        except CancelledError:
            print("Cancelled.")
            return 0
        except End:
            return 0

main()