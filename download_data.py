import objaverse
import pandas as pd
import objaverse
import pandas as pd
# pip install objaverse pandas
import multiprocessing
kiui_uids = pd.read_csv("kiuisobj_v1_merged_80K.csv", header=None)
processes = multiprocessing.cpu_count()
uids = kiui_uids[1].values.tolist()
# make sure you have enough disk capacity
objaverse.load_objects(uids, download_processes=processes)