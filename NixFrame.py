import nixio as nix
import numpy as np
import os
import pandas as pd
import pickle
import helperfunctions as hf
from copy import deepcopy

'''
This File is meant to convert .nix Files into a more readable format (pandas.DataFrame)

NixToFrame(<folder>) searches all folders in <folder> for .nix files and converts them with 
DataFrame(<nixfile>) and saves them as picklefile

df = load_data(<filename>) can load these picklefiles
'''


def DataFrame(nixfile, before=0.001, after=0.001, savefile=False, saveto='./', mindepth=0, delimiter=';'):
    '''
    opens a nix file, extracts the data and converts it to a pandas.DataFrame

    :param nixfile (string): path and name of .nix file
    :param before (float): time before sweeps that is saved, too (in the unit that is given in the file. Usually [s])
    :param after (float): time after sweeps that is saved, too (in the unit that is given in the file. Usually [s])
    :param savefile (string, bool): if not False and a string, the dataframe will be saved as <savefile>.pickle, if True it will be saved as nixfile.split('.nix')[0]
    :param saveto (string): path to save the files in path defined by saveto
    :param mindepth (int): minimum number of nested entries in dataframe
    :param delimiter (string): internally used to create the nested dataframe, use a delimiter that does not naturally occur in your protocol names

    :return dataframe (pandas.DataFrame): pandas.DataFrame with available nix data
    '''

    try:
        block = nix.File.open(nixfile,'r').blocks[0]
    except:
        print('cannot open ' + nixfile.split('/')[-1] + ', skip it')
        return None

    data_arrays = block.data_arrays
    names = [data_arrays[i].name for i in range(len(data_arrays))]
    shapes = [x.shape for x in data_arrays]
    data_names = np.array([[x, i] for i, x in enumerate(names) if (shapes[i][0] >= 0.999 * shapes[0][0])])
    header_names = [mt.name for mt in block.multi_tags]

    try:
        data_traces = np.array([data_arrays[name][:] for name, idx in data_names])
    except:
        print(nixfile.split('/')[-1] + ' has corrputed data_arrays, no file will be saved')
        return None

    time = data_arrays[1].dimensions[0].axis(data_arrays[1].shape[0])
    dt = time[1]-time[0]

    block_metadata = {}
    block_metadata[block.id] = GetMetadataDict(block.metadata)

    tag = block.tags
    tag_metadata = {}
    tag_id_times = {}
    protocols = [[]] * len(tag)
    for i,t in enumerate(tag):
        protocols[i] = {}
        protocols[i]['id'] = t.id
        protocols[i]['name'] = t.name
        protocols[i]['position'] = t.position[0]
    protocols = pd.DataFrame(protocols)

    for i in range(len(tag)):
        try:
            meta = tag[i].metadata
            tag_metadata[meta.id] = GetMetadataDict(meta)
            tag_id_times[meta.id] = [tag[i].position[0], tag[i].position[0]+tag[i].extent[0]]
        except:
            print(nixfile.split('/')[-1] + ' has no tags, no file will be saved')
            return None

    protocol_idcs_old = np.where([(' onset times' in name) for name in names])[0]
    durations_idcs_old = np.where([(' durations' in name) for name in names])[0]
    for idx in protocol_idcs_old:
        names[idx] = names[idx].split(' onset times')[0] + '_onset_times'
    for idx in durations_idcs_old:
        names[idx] = names[idx].split(' durations')[0] + '_durations'

    protocol_idcs = np.where([('_onset_times' in name) for name in names])[0]

    if len(protocol_idcs) == 0:
        print(nixfile.split('/')[-1] + ' is empty, no file will be saved')
        return None

    data_df = []
    for hn in header_names:
        idcs = np.where(np.array([hn in name[:len(hn)] for name in names], dtype=int))[0]
        for j,idx in enumerate(idcs):
            data = data_arrays[int(idx)][:]
            if j == 0:
                i0 = len(data_df)
                protname = hn.split('-')[0]
                typename = hn.split('-')[1]
                if protname == 'VC=':
                    continue

                data_df.extend([[]]*len(data))
                for jj in range(len(data)):
                    data_df[i0 + jj] = {}
                    data_df[i0 + jj]['type'] = typename
                    # data_df[i0 + jj]['row_id'] = uuid.uuid4()
                    if typename == 'PNSubatraction':
                        data_df[i0 + jj]['type'] = 'PNSubtraction'
                    elif typename == 'Qualitycontrol':
                        data_df[i0 + jj]['type'] = 'QualityControl'

            dataname = names[idx].split(hn)[1][1:]
            if protname == 'VC=':
                continue
            if dataname == 'Current-1':
                continue
            for jj in range(len(data)):
                data_df[i0 + jj][dataname] = data[jj]

    keys = np.unique([list(d.keys()) for d in data_df])
    k = []
    for key in keys:
        k.extend(key)
    keys = np.unique(k)
    if 'repro_tag_id' not in keys:
        for i in range(len(data_df)):
            data_df[i]['repro_tag_id'] = protocols[~(protocols.position >= data_df[i]['onset_times'])]['id'].iloc[-1]

    traces_idx = np.where([("data.sampled" in d.type) or ("data.events" in d.type) for d in data_arrays])[0]
    traces_df = []


    for i,idx in enumerate(traces_idx):
        for j in range(len(data_df)):
            if i == 0:
                traces_df.append({})

            if "data.sampled" in data_arrays[names[int(idx)]].type:
                idx0 = int((data_df[j]['onset_times'] - before) / dt)
                idx1 = int((data_df[j]['onset_times'] + data_df[j]['durations'] + after) / dt)
                if idx0>=idx1:
                    traces_df[j][names[idx]] = np.array([np.nan])
                else:
                    traces_df[j][names[idx]] = data_arrays[names[int(idx)]][idx0:idx1]
            elif "data.events" in data_arrays[names[int(idx)]].type:
                idx0 = int((data_df[j]['onset_times'] - before) / dt)
                idx1 = int((data_df[j]['onset_times'] + data_df[j]['durations'] + after) / dt)
                t0 = data_df[j]['onset_times'] - before
                t1 = data_df[j]['onset_times'] + data_df[j]['durations'] + after
                if t0>=t1:
                    traces_df[j][names[idx]] = np.array([np.nan])
                else:
                    arr = data_arrays[names[int(idx)]][:]
                    traces_df[j][names[idx]] = arr[(arr>=t0) & (arr<=t1)] - data_df[j]['onset_times']

            if i == 0:
                traces_df[j]['time'] = time[idx0:idx1] - data_df[j]['onset_times']
                traces_df[j]['time_before_stimulus'] = before
                traces_df[j]['time_after_stimulus'] = after
                traces_df[j]['samplingrate'] = 1 / dt
                traces_df[j]['meta_id'] = list(block_metadata.keys())[0]

                if type(data_df[j]['repro_tag_id']) == bytes:
                    data_df[j]['repro_tag_id'] = data_df[j]['repro_tag_id'].decode("utf-8")
                traces_df[j]['protocol'] = np.array(protocols[protocols.id == data_df[j]['repro_tag_id']].name)[0].split('_')[0]
                traces_df[j]['protocol_number'] = np.array(protocols[protocols.id == str(data_df[j]['repro_tag_id'])].name)[0].split('_')[1]
                traces_df[j]['id'] = data_df[j]['repro_tag_id']

    metadic = {}
    for i,key in enumerate(protocols.id):
        d = GetMetadataDict(tag[key].metadata)
        if (len(d.keys()) == 1) and ('RePro-Info' in list(d.keys())[0]):
            d = d['RePro-Info']
        metadic[key] = DicCrawl(d, key='tag_meta', delimiter=delimiter)
        metadic[key].update(DicCrawl(block_metadata[list(block_metadata.keys())[0]], key='block_meta', delimiter=delimiter))
    meta_df = [[]]*len(data_df)
    for i in range(len(data_df)):
        meta_df[i] = metadic[str(data_df[i]['repro_tag_id'])]

    dics = [meta_df, data_df, traces_df]
    old_maxcount = mindepth

    for k in range(len(dics)):
        for j in range(len(dics[k])):
            keys = list(dics[k][j].keys())

            counts = np.array([key.count(delimiter) for key in keys])
            maxcount = np.max(counts)
            if maxcount < mindepth:
                maxcount = mindepth

            if maxcount < old_maxcount:
                maxcount = old_maxcount

            # append delimiters to the keys until each key contains the same number of keys
            for i, key in enumerate(keys):
                add = maxcount - counts[i]
                newkey=key
                for ii in range(add):
                    newkey += delimiter
                dics[k][j][newkey] = dics[k][j].pop(key)
            old_maxcount = maxcount

    data_df = pd.DataFrame(data_df)
    traces_df = pd.DataFrame(traces_df)
    meta_df = pd.DataFrame(meta_df)

    data = pd.concat([meta_df, data_df, traces_df], axis=1, sort=False)
    data.columns = data.columns.str.split(';', expand=True)

    if savefile != False:
        if savefile == True:
            savefile = nixfile.split('.nix')[0]
        if saveto != None:
            savefile = savefile.split('/')[-1]
            savefile = saveto+savefile

        with open(savefile + '_dataframe.pickle', 'wb') as f:
            pickle.dump(data, f, -1)  # create pickle-files, using the highest pickle-protocol
    return data


def NixToFrame(folder, before=0.0, after=0.0, skipold=True, saveto=None, mindepth=0):
    '''
    searches subfolders of folder to convert .nix files to a pandas dataframe and saves them in the folder

    :param folder: path to folder that contains subfolders of year-month-day-aa style that contain .nix files
    :param skipold (bool): skip creation of dataframe if it already eists
    :param saveto (string): path where files should be saved, if None it saves in the folder, where the .nix file is
    '''
    if folder[-1] != '/':
        folder = folder + '/'

    dirlist = os.listdir(folder)
    for dir in dirlist:
        if os.path.isdir(folder + dir):
            for file in os.listdir(folder+dir):
                if ('.nix' in file):
                    if skipold == True:
                        if np.sum(['_dataframe.pickle' in x for x in os.listdir(folder+dir)]) >= 1:
                            print('skip ' + file)
                            continue
                    print(file)
                    DataFrame(folder+dir+'/'+file, before, after, True, saveto, mindepth=mindepth)


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)  # load data with pickle
    return data


def GetMetadataDict(metadata):
    def unpackMetadata(sec):
        metadata = dict()
        metadata = {prop.name: sec[prop.name] for prop in sec.props}
        if hasattr(sec, 'sections') and len(sec.sections) > 0:
            metadata.update({subsec.name: unpackMetadata(subsec) for subsec in sec.sections})
        return metadata
    return unpackMetadata(metadata)


def DicCrawl(dic, key='', delimiter=';'):
    keylist = GetKeyList(dic, key=key, delimiter=delimiter)
    newdic = KeylistToDic(keylist)
    return newdic


def GetKeyList(dic, keylist=[], key='', delimiter=';'):
    items = dic.items()
    for item in items:
        if type(item[1]) == dict:
            if len(key) == 0:
                keylist = GetKeyList(item[1], keylist, str(item[0]), delimiter)
            else:
                keylist = GetKeyList(item[1], keylist, key + delimiter + str(item[0]), delimiter)
        else:
            if len(key) == 0:
                keylist.append([str(item[0]), item[1]])
            else:
                keylist.append([key + delimiter + str(item[0]), item[1]])
    return keylist


def KeylistToDic(keylist):
    dic = {}
    for item in keylist:
        dic[item[0]] = item[1]
    return dic


def PNSubtraction(df, currenttrace='Current-1', newcurrenttrace='Current-2', kernelcurrenttrace='Current-3', delimiter=';'):
    '''
    Only for VoltageClamp experiments and WILL BE OBSOLETE, SOON

    :param df (pandas.DataFrame): NixFrame.DataFrame
    :param currenttrace:
    :param newcurrenttrace:
    :param kernelcurrenttrace:
    :param delimiter:
    :return:
    '''

    df = deepcopy(df)
    if 'TraceId' not in df.columns:
        for id in np.unique(df.repro_tag_id):
            dfp = df[df.repro_tag_id == id]
            if np.isnan(dfp.pn.iloc[0]) or dfp.pn.iloc[0] == 0:
                continue

            pn = int(dfp.pn.iloc[0])
            for i in np.arange(0,len(dfp),np.abs(pn)+1, dtype=int):
                locs_pn = df[df.repro_tag_id == id].iloc[i:i+np.abs(pn)].index
                loc = locs_pn[-1] + 1

                if loc <= df[df.repro_tag_id == id].index[-1]:
                    df.loc[locs_pn, 'type'] = 'PNSubtraction'
                    df.loc[loc, 'type'] = 'Trace'


    trace_idx = np.where(df.type == 'Trace')[0]
    currentdic = [[]] * len(trace_idx)
    kerneldic = [[]] * len(trace_idx)
    affix = ''
    affix_num = len(list(df)[0]) - 1
    for i in range(affix_num):
        affix += delimiter

    for idx,i in enumerate(trace_idx):
        if np.isnan(df.pn.iloc[i]) or df.pn.iloc[i] == 0:
            continue

        if 'TraceId' in df.columns:
            trace_id = df.TraceId.iloc[i]
            idxvec = np.where((df.TraceId == trace_id) & (df.type == 'PNSubtraction'))[0]
        else:
            delays = (df['onset_times'][df.type == 'PNSubtraction']) - df['onset_times'].iloc[i]
            maxidx = delays[np.array(delays)<0].index[-1]
            idxvec = np.arange(maxidx-np.abs(df.pn.iloc[i])+1, maxidx+.1, dtype=int)

        pn_traces = hf.get_traces(currenttrace, df.iloc[idxvec])
        I_trace = np.array(df[currenttrace].iloc[i])

        pn_traces = np.mean(pn_traces, axis=1)
        pn_traces -= np.mean(pn_traces[:int(df.iloc[i].time_before_stimulus * df.iloc[i].samplingrate)])
        pn_trace = pn_traces*df.pn.iloc[i]
        I_trace -= np.mean(I_trace[:int(df.iloc[i].time_before_stimulus * df.iloc[i].samplingrate)])

        ''' try kernels '''
        Vpn_trace = hf.get_traces('V-1', df.iloc[idxvec])
        tpn = hf.get_traces('time', df.iloc[idxvec])[:, 0]
        Vtr_trace = np.array(df['V-1'].iloc[i])
        ttr = np.array(df['time'].iloc[i])
        kernel_length = 3
        idx0 = np.nanargmin(np.abs(tpn)) - 1
        idx1 = np.nanargmin(np.abs(tpn - kernel_length / 1000))
        idx40 = np.nanargmin(np.abs(tpn - 0.04))
        idx60 = np.nanargmin(np.abs(tpn - 0.06))

        Vpn = tpn * 0.0 + np.mean(np.mean(Vpn_trace, axis=1)[:idx0])
        Vpn[tpn >= 0] = np.mean(np.mean(Vpn_trace, axis=1)[idx40:idx60])
        Vtr = ttr * 0.0 + np.mean(Vtr_trace[:idx0])
        Vtr[ttr >= 0] = np.mean(Vtr_trace[idx40:idx60])

        ftinput = np.fft.fft(np.diff(Vpn[idx0:idx1]))
        ftoutput = np.fft.fft(pn_traces[idx0:idx1 - 1])
        kernel = np.fft.ifft(ftoutput / ftinput)
        kernel = np.append(kernel, np.zeros(len(I_trace) - len(kernel))+np.mean(kernel[-20:]))
        kerneldic[idx] = {}
        kerneldic[idx][kernelcurrenttrace + affix] = I_trace - np.convolve(np.diff(Vtr), kernel)[:len(I_trace)] #- np.convolve(np.diff(Vtr),kernel2)[:len(I_trace)]
        ''' end of kernels '''


        if len(pn_trace) < len(I_trace):
            I_trace = I_trace[:len(pn_trace)]
        elif len(pn_trace) > len(I_trace):
            pn_trace = pn_trace[:len(I_trace)]

        currentdic[idx] = {}
        currentdic[idx][newcurrenttrace+affix] = I_trace - pn_trace

    currentdf = pd.DataFrame(currentdic, index=df[df.type == 'Trace'].index)
    currentdf.columns = currentdf.columns.str.split(';', expand=True)
    df = pd.concat([df, currentdf], axis=1)
    df = df.drop(index = df[df.type == 'PNSubtraction'].index)
    return df


def QualityControl(df, currenttrace='Current-1', potentialtrace='V-1', delimiter=';'):
    '''
    only for VoltageClamp experiments
    '''

    qc = [[]]*len(df[df.type == 'QualityControl'])
    prefix = 'qualitycontrol' + delimiter
    affix = ''
    affix_num = len(list(df)[0]) - 2
    for i in range(affix_num):
        affix += delimiter

    # assume that stimulus has 10ms at holdingpotential followed by 10ms at holdingpotential-20
    samplingrate = df.samplingrate.iloc[0]
    idx10ms = int(0.01 * samplingrate)
    idx20ms = int(0.02 * samplingrate)
    idx1ms = int(0.001 * samplingrate)

    for i in range(len(qc)):
        qc[i] = {}
        qc[i][prefix + currenttrace + affix] = df[df.type == 'QualityControl'][currenttrace].iloc[i]
        qc[i][prefix + potentialtrace + affix] = df[df.type == 'QualityControl'][potentialtrace].iloc[i]
        qc[i][prefix + 'onset_times' + affix] = df[df.type == 'QualityControl']['onset_times'].iloc[i]
        qc[i][prefix + 'holdingpotential' + affix] = df[df.type == 'QualityControl']['tag_meta']['settings']['holdingpotential'].iloc[i]
        if 'TraceId' in df.columns:
            qc[i][prefix + 'TraceId' + affix] = df[df.type == 'QualityControl']['TraceId'].iloc[i]

        I1 = np.mean(df[df.type == 'QualityControl'][currenttrace].iloc[i][idx10ms-idx1ms : idx10ms - 2])
        I2 = np.mean(df[df.type == 'QualityControl'][currenttrace].iloc[i][idx20ms-idx1ms : idx20ms - 2])
        R = 20/(I1-I2)
        qc[i][prefix + 'resistance' + affix] = R
    qc_df = pd.DataFrame(qc)

    if len(qc) == 0:
        return df

    qc_empty = [{}]
    for key in qc[0].keys():
        qc_empty[0][key] = np.nan
    qc_dic = qc_empty * len(df)

    for idx in df[df.type == 'Trace'].index:
        i = np.where(df.index == idx)[0][0]
        if 'TraceId' in df.columns:
            qc_dic[i] = qc[np.where(qc_df[prefix + 'TraceId' + affix] == df.TraceId.loc[idx])[0][0]]
        else:
            None
        delays = qc_df[prefix + 'onset_times' + affix] - df['onset_times'].loc[idx]
        maxidx = np.where(delays[np.array(delays) < 0])[0][0]
        qc_dic[i] = qc[maxidx]

    qc = pd.DataFrame(qc_dic, index=df.index)
    qc.columns = qc.columns.str.split(';', expand=True)

    df = pd.concat([df, qc], axis=1)
    df = df.drop(index=df[df.type == 'QualityControl'].index)
    return df

