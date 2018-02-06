from __future__ import print_function
from __future__ import absolute_import

import joblib
import os 
import shutil
from rllab.misc import logger
from rllab.misc.console import colorize


# Output handling
# ---------------

OUTPUT_DIR = None
FINAL_OUTPUT_DIR = None
ENSURE_CLEAN_OUTPUT_DELETE_ALL = False

def set_delete_all(value):
    global ENSURE_CLEAN_OUTPUT_DELETE_ALL
    ENSURE_CLEAN_OUTPUT_DELETE_ALL = value

def ensure_clean_output_dir(output_dir):
    global ENSURE_CLEAN_OUTPUT_DELETE_ALL    
    if os.path.exists(output_dir):
        if ENSURE_CLEAN_OUTPUT_DELETE_ALL:
            shutil.rmtree(output_dir)
        else:
            while True:
                text = raw_input("Delete contents of '%s' ('yes' or 'no' or 'all')? " % (output_dir,))
                if text == 'all':
                    ENSURE_CLEAN_OUTPUT_DELETE_ALL = True
                    shutil.rmtree(output_dir)
                    break
                if text == 'yes':
                    shutil.rmtree(output_dir)
                    break
                elif text == 'no':
                    print("Output directory is not clear. Old files will not be overwritten.")
                    break
                    #raise Exception("Need clear output directory to proceed.")
    else: 
        os.makedirs(output_dir)

def setup_output(output_dir, clean=True, final_output_dir=None):
    global OUTPUT_DIR
    global FINAL_OUTPUT_DIR
    if OUTPUT_DIR is not None:
        shutdown_output()
    output_dir = os.path.abspath(output_dir)
    if clean:
        ensure_clean_output_dir(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    OUTPUT_DIR = output_dir
    FINAL_OUTPUT_DIR = final_output_dir

    print("** Output set to", OUTPUT_DIR)
    if FINAL_OUTPUT_DIR is not None:
        print("** Final output set to", FINAL_OUTPUT_DIR)
 
    logger.add_text_output(os.path.join(OUTPUT_DIR, "rllab.txt"))
    logger.add_tabular_output(os.path.join(OUTPUT_DIR, "rllab.csv"))
    logger.set_snapshot_mode('all')     # options: 'none', 'last', or 'all'
    logger.set_snapshot_dir(OUTPUT_DIR)

def shutdown_output():
    global OUTPUT_DIR
    assert OUTPUT_DIR is not None, "Cannot shutdown output that has not been setup."
    logger.remove_text_output(os.path.join(OUTPUT_DIR, "rllab.txt"))
    logger.remove_tabular_output(os.path.join(OUTPUT_DIR, "rllab.csv"))

    OUTPUT_DIR = None

def write_object(filename, obj):
    #assert OUTPUT_DIR is not None, "Output directory must be set."
    if OUTPUT_DIR is None:
        print("No output directory set but attempted to write to " + filename)
    else:
        outname = os.path.join(OUTPUT_DIR, filename)
        if not os.path.isfile(outname):
            joblib.dump(obj, os.path.join(OUTPUT_DIR, filename), compress=3, protocol=2)
        else:
            print( "Failed to write " + outname + ". File already exists.")    
    # with open(os.path.join(OUTPUT_DIR, filename), 'wb') as f:
    #     pickle.dump(obj, f, protocol=2)

def read_object(filename):
    if OUTPUT_DIR is None:
        print("No output directory set but attempted to read " + filename)
    else:
        # first try OUTPUT_DIR then FINAL_OUTPUT_DIR
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.isfile(path):
            if FINAL_OUTPUT_DIR is not None:
                path = os.path.join(FINAL_OUTPUT_DIR, filename)
                if not os.path.isfile(path):
                    raise Exception("File %s cannot be found in either output dir (%s) or final output dir (%s)"
                        % (filename, OUTPUT_DIR, FINAL_OUTPUT_DIR))
            else:
                raise Exception("File %s cannot be found in output dir (%s)"
                    % (filename, OUTPUT_DIR))
        obj = joblib.load(path)
    return obj

def exists_object(filename, output_dir=None):
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    if output_dir is None:
        return False
    else:
        outname = os.path.join(output_dir, filename)
        return os.path.isfile(outname)

def get_output_dir():
    return OUTPUT_DIR
