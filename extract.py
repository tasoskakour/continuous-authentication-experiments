# pylint: disable = C0111, C0103, C0411, C0301, W0102, C0330
"""This module is used to extract the key_holds times and digraph
   up_down times from the raw events of the subjects."""
import read_write
import numpy as np
import operator
import time
import general_purpose

# All limits are in milliseconds
KEY_HOLD_UPPER_LIMIT = 400
KEY_HOLD_LOWER_LIMIT = 0
DIGRAPH_UP_DOWN_UPPER_LIMIT = 950
DIGRAPH_UP_DOWN_LOWER_LIMIT = -400


def _my_search_event(eventlist, event, key=''):
    """Searches a list of events for specific type of event and the specific key if specified"""
    if key == '':
        for i, val in enumerate(eventlist):
            if val['event'] == event:
                return i, val
    else:
        for i, val in enumerate(eventlist):
            if val['event'] == event and val['key'] == key:
                return i, val
    # print 'Returning -1 from my searchevent', event, key
    return -1, {}


def _digraph_all(subject_events_data, ignore_space=False, sortByDigraph=True):
    """Extracts the subject's digraph timings of key_holds and up_down by the raw events.
    Parameters
    ----------
    \n `subject_events_data` (list) The raw events list
    \n `ignore_space` (boolean) If True it ignores space
    \n `sortByDigraph` (boolean) If True it sorts data by digraph
    Returns
    ---------
    \n A list of dicts  [{'digraph', 'points'}]
    where points is a nx3 numpy array with x,y,z as key_hold_1, key_hold_2 and up_down timings of the digraph
    """
    ret = []
    # work with a copy because the pop method changes the list of dict :/
    events = subject_events_data[:]

    if ignore_space is True:
        events = [evt for evt in events if evt['data'] != 'Space']

    while True:
        if len(events) <= 2:
            break

        # The next keyDown event will be the first
        key_1_down_event = events[0]
        if key_1_down_event['event'] != 'keystrokeDown':
            print '...digraph_all: Continuing, first event is not keydown ->', events[0]
            events.pop(0)
            continue

        # Find the respective keyUp event of key_1
        key_1_up_event_index, key_1_up_event = _my_search_event(
            events[1:], 'keystrokeUp', key_1_down_event['key'])
        if key_1_up_event_index == -1:
            print '...digraph_all: Continuing, Couldnt find keystrokeUp event for key = ' + str(key_1_down_event)
            events.pop(0)
            continue
        else:
            key_1_up_event_index += 1

        # Find the following keyDown event after the keyDown of key_1
        key_2_down_event_index, key_2_down_event = _my_search_event(
            events[1:], 'keystrokeDown')
        if key_2_down_event_index == -1:
            print '1993: What now?'
        else:
            key_2_down_event_index += 1

        # Find the respective keyUp event of key_2
        key_2_up_event_index, key_2_up_event = _my_search_event(
            events[key_2_down_event_index + 1:], 'keystrokeUp', key_2_down_event['key'])
        if key_2_up_event_index == -1:
            events.pop(0)
            events.pop(key_1_up_event_index - 1)  # index has changed now
            print '1994: Removed Noise due to missing keyUp for key: ' + str(key_2_down_event)
            continue
        else:
            key_2_up_event_index += key_2_down_event_index + 1

        # Calculate
        # Here if I want down_down: "down_down": key_2_down_event['timestamp'] - key_1_down_event['timestamp'],
        digraph_obj = {
            "digraph": key_1_down_event['key'] + key_2_down_event['key'],
            "up_down": key_2_down_event['timestamp'] - key_1_up_event['timestamp'],
            "key_holds": [key_1_up_event['timestamp'] - key_1_down_event['timestamp'], key_2_up_event['timestamp'] - key_2_down_event['timestamp']]
        }
        xyz = np.array([[digraph_obj['key_holds'][0],
                         digraph_obj['key_holds'][1], digraph_obj['up_down']]])
        # Store appropriately
        if (general_purpose.is_not_extreme_outlier(digraph_obj['up_down'], DIGRAPH_UP_DOWN_LOWER_LIMIT, DIGRAPH_UP_DOWN_UPPER_LIMIT)
                and general_purpose.is_not_extreme_outlier(digraph_obj['key_holds'][0], KEY_HOLD_LOWER_LIMIT, KEY_HOLD_UPPER_LIMIT)
                and general_purpose.is_not_extreme_outlier(digraph_obj['key_holds'][1], KEY_HOLD_LOWER_LIMIT, KEY_HOLD_UPPER_LIMIT)
                ):
            if ret == []:
                ret.append({"digraph": digraph_obj['digraph'],
                            "points": xyz})
            else:
                tmpi = -1
                for i, val in enumerate(ret):
                    if val['digraph'] == digraph_obj['digraph']:
                        tmpi = i
                        break
                if tmpi != -1:
                    ret[tmpi]['points'] = np.append(
                        ret[tmpi]['points'], xyz, axis=0)
                else:
                    ret.append({"digraph": digraph_obj['digraph'],
                                "points": xyz})
        # Update and remove the 1st key down and up for next iteration
        events.pop(0)
        events.pop(key_1_up_event_index - 1)  # index has changed now

    # Sort by Digraph
    if sortByDigraph is True:
        ret = sorted(ret, key=operator.itemgetter('digraph'))
    return ret


def one(doc, ignore_space=False, logg=True):
    """Extracts digraph up_down and key_holds times from subject doc events
    Returns
    --------
    \nAn Object with '_subject', '_track_code' and 'data': as calculated from `_digraph_all` function
    """
    start = time.time()
    ret = {
        "_subject": doc['subject'], "_track_code": doc['track_code'],
        "data": _digraph_all(doc['sessions']['data'], ignore_space=ignore_space, sortByDigraph=True)}
    if logg is True:
        print '-Subject Timings of "' + doc['subject'] + '" extracted in ' + str(time.time() - start) + ' seconds.'
    return ret


def all(docs, write_to_json=False, ignore_space=False, filename='subjects-data'):
    """Just some wrapper that takes all docs\n
    Returns
    --------
    \nA list -> [{'_subject': '', '_track_code': '', data: {[...]]}}]
    """
    start = time.time()
    ret = []
    for subject_doc in docs:
        ret.append(one(subject_doc, ignore_space=ignore_space))
    if write_to_json is True:
        read_write.write_timings_to_local(ret, filename)
    print '-All timings extracted in: %.2f s.' % (time.time() - start)
    return ret
