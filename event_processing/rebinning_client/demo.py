import base64
from dataclasses import dataclass, asdict
import io
import json
import math
import pathlib
import re
from typing import Any, Optional
from fastapi.responses import StreamingResponse
from nicegui import run, ui, app, Client
import numpy as np
import plotly.graph_objects as go
import requests
import uuid

from event_processing.rebinning_api import client as binning_client, models as binning_models

all_instruments = requests.get(url="https://ncnr.nist.gov/ncnrdata/metadata/api/v1/instruments").json()
available_instrument_names = [
    "vsans",
    "macs",
    "candor",
]

FAVICON_DATA = """\
iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAABGdBTUEAALGPC/xhBQAAACBjSFJN
AAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAACFlBMVEUAAAAAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkAeOkA
eOkAeOn///+5vaZfAAAAsHRSTlMAASpVZmxhPg0bi+b6uUtu9Bma5CaI/sSdj67r4RVGvywKcbMG
2gf5T2MRf9HMXAXgOx7sfZId27ZTJ59UDuh+NsqnWKw0ZZbXjBTjhAvvhwP7KFuqoHVXgoD1z213
iRDpe2uR3c0CXy3llUi8RLoiIBbtq2+DoviGaLDLnM43wJMXkHP20BpwQcJNrxPBsmD9QnyXpTP8
Eh8lUAjS1O7Yt1IplHmOWXLxMiNaxdzTSkV9vKAAAAABYktHRLE0Y55SAAAAB3RJTUUH6AEXCxkA
UTgzIAAAAsJJREFUWMPtlelbElEUxgfLUFETLRJbMaGUCinFxLTSNCRJSUuzlTLS9lXLTNt3Wy1b
rGzfa/7EeO8d5t4r0/PM9HnOBznnPe/5MVyOF0kyFpa0adPTZ1gzDI4lIzPLJpPIzsn9n/mZeTKL
HLvRcUu+LETBLIOA2WTMMafQWTSXpPPmG5pf4MDQwkXIXcXZKBYbmS9xY6TIk8QtQbnUAKAUA7Yy
tfaiXmbRD1iOgRWs9pVD8OueX4kTWOXhlAoAsnQD0mGv5JUAlCrdgNWwVwtSFaSgXkAB3OLu1UBa
o3O+FkdQJ2rFAKzVCVgH83pRq4ZWrxPQAPMGUWvUgP4zmmDeKGrkawjpBJDP2yxqYWibdALI1rSI
WgTaZp2AVpgrRK0NWjTFuqW9Nb9ja/O2zgCvdsG8XXR6oHVLHTToloe9bnbj7NjpUs27IOwWAXug
7ZUUdySh+GL8nYf/XnV396HcLwLs0Ho4QOCAnBLOODUfRFEjAoLkRmCA3j5ZIw5R82HkR0TAUWjH
VMBxm6wZTcR8AqlbBJTSj6X4Tobo66nTztiZs/3safIyYY53J9KBEgFwDu3zSUA9+Tvob6NNi/VC
kjBEhItIhwXACCSrxD1tOX/DxS8p6sBllFeQXuXnfXioaxEOcP2G8AaWfkW/ieoWstt8/w6UuxID
9E39nblHf3/kUbK3pEjj2vchPGCAh49StvqxcozkXJ4gHWPNp8/QGmcAjavluYO2yPu+iCJ9mey9
miB7JjHAcCpAek1b7aSoJCfaSTtlo6jehBlgQmNeGqM9Lyky3iJ3TL57b/d30eMhV7IC+KAF+Eh7
n5Tdr5uypBSsFJ+1AE20V6iUwRA/Hv0i8YBGLUAD7am3savlqzo/8k0SAN+1ALm0N8mU2h9DP3/1
/B6M9aqSAhjXAvxJAWiEAoiYABNgAkyACTABifgL0UQOtt15DR4AAAAldEVYdGRhdGU6Y3JlYXRl
ADIwMjQtMDEtMjNUMTY6MjU6MDAtMDU6MDAYAw6eAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI0LTAx
LTIzVDE2OjI1OjAwLTA1OjAwaV62IgAAAABJRU5ErkJggg==
""".replace("\n", "")

FAVICON = f"data:image/png;base64,{FAVICON_DATA}"
NCNR_METADATA_API_URL = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1"
# CHRNS_REBINNING_API_URL = "http://localhost:8000"
# TODO: convert to hidden form with post urlencoded, so that we don't have
# to look up state (can send the bin edges in post)
SESSION_LOOKUP = {}

static_folder = pathlib.Path(__file__).parent / 'static'
app.add_static_files('/static', static_folder)

@app.get('/download/nexus/{session_id}')
async def download_nexus(session_id: str):
    session_data = SESSION_LOOKUP.get(session_id, None)
    if session_data is None:
        return
    
    binning_state = session_data['binning_state']
    time_bin_settings = session_data['time_bin_settings']

    metadata: binning_models.MetadataReply = binning_state['metadata']
    if metadata is None:
        return
    bins = time_bin_settings.get_rebin_time_bins_object()
    binning_state['downloading_nexus'] = True
    print("starting nexus rebin")
    nexus_reply = await run.io_bound(binning_client.get_nexus, metadata, bins)
    print("done with nexus rebin")
    orig_filename = metadata.measurement.filename
    orig_path = pathlib.Path(orig_filename)
    file_suffixes = ''.join(orig_path.suffixes)
    file_stem = re.sub(f"{file_suffixes}$", '', orig_filename)
    new_filename = f"{file_stem}_rebinned{file_suffixes}"
    nexus_bytes = base64.b64decode(nexus_reply.base64_data)
    result = io.BytesIO(nexus_bytes)
    buffer_size = 2**16 # 64K
    async def result_streamer():
        buffer = result.read(buffer_size)
        while buffer:
            yield buffer
            buffer = result.read(buffer_size)

        binning_state['downloading_nexus'] = False
        
    headers = {
        'Content-Disposition': f'attachment; filename="{new_filename}"',
        'Content-Type': 'application/hdf5',
        'Content-Length': f'{len(nexus_bytes)}',
    }
    return StreamingResponse(result_streamer(), headers=headers)


@ui.page('/')
async def index(client: Client):
    """ per-connection state defined within """
    session_id = uuid.uuid4().hex
    SESSION_LOOKUP[session_id] = {}

    experiment_search_state = {
        "rows": [],
        "pagination": {
            "rowsPerPage": 10,
            "descending": True,
            "sortBy": "date",
            "page": 0,
            "rowsNumber": 0,
        },
        'selection': '',
    }

    datafile_search_state = {
        'rows': [],
        "pagination": {
            "rowsPerPage": 10,
            "descending": True,
            "sortBy": "date",
            "page": 0,
            "rowsNumber": 0,
        },
        'selection': '',
        'selection_path': '',
    }

    binning_state = {
        'fetching_summary': False,
        'downloading_nexus': False,
        'duration': None,
        'num_bins': None,
        'bins': None,
        'metadata': None,
        'last_frame': None
    }

    SESSION_LOOKUP[session_id]['binning_state']  = binning_state

    @ui.refreshable
    def experiments_table():
        experiment_columns = [
            {'name': 'experiment_id', 'label': 'Experiment ID', 'field': 'id', 'required': True, 'align': 'left', 'style': 'width: 8em;'},
            {'name': 'title', 'label': 'Title', 'field': 'title', 'align': 'left', 'style': 'max-width:2px;'},
            {'name': 'participants', 'label': 'Participants', 'field': 'participant_names', 'align': 'left', ':format': 'value => JSON.parse(value).join(", ")', 'style': 'max-width:2px;'},
        ]
        def select_handler(e):
            selected_id = e.selection[0]['id'] if len(e.selection) > 0 else ''
            experiment_search_state['selection'] = selected_id
            if selected_id != '':
                ui.tab_panels(tabs).value = file_selector

        table = ui.table(
            columns=experiment_columns,
            rows=experiment_search_state['rows'],
            row_key='id',
            pagination=experiment_search_state['pagination'],
            selection="single",
            on_select=select_handler,
        )
        table.on('request', experiment_search_params.pagination_request_handler)
        table.classes("table-fixed w-full")
        table.props('wrap-cells=true')
        return table

    def get_metadata_columns(rows):
        keys = set()
        for row in rows:
            metadata_string = row.get('metadata', '')
            metadata = json.loads(metadata_string) if metadata_string else {}
            row['metadata'] = metadata
            for key in metadata:
                keys.add(key)
        
        extra_cols = []
        for key in keys:
            col_def = {'name': key, 'label': key, ':field': f'row => row.metadata["{key}"]'}
            extra_cols.append(col_def)
        return extra_cols

    @ui.refreshable
    def datafiles_table():
        columns = [
            { 'name': 'filename', 'label': 'Filename', 'field': 'filename', 'required': True },
            { 'name': 'cycle', 'label': 'Rx Cycle', 'field': 'rxcycle_id' },
            { 'name': 'start_date', 'label': 'Start Date', 'field': 'start_date' },
        ]

        def select_handler(e):
            if len(e.selection) == 0:
                selected_id = ''
                selected_path = ''
            else:
                selected = e.selection[0]
                selected_id = selected['filename']
                selected_path = selected['localdir']
            datafile_search_state['selection'] = selected_id
            datafile_search_state['selection_path'] = selected_path
            if selected_id != '':
                ui.tab_panels(tabs).value = rebinning_params
                get_metadata()
        
        extra_cols = get_metadata_columns(datafile_search_state['rows'])

        table = ui.table(
            columns=columns + extra_cols,
            rows=datafile_search_state['rows'],
            row_key='filename',
            pagination=datafile_search_state['pagination'],
            selection="single",
            on_select=select_handler,
        )
        table.on('request', datafile_search_params.pagination_request_handler)
        table.classes("table-fixed w-full sticky-selector-column")
        table.props('wrap-cells=true')
        return table

    @dataclass
    class ExperimentSearchParams:
        instrument_name: Optional[str] = None
        participant_name: Optional[str] = None
        experiment_title: Optional[str] = None
        experiment_id: Optional[str] = None # '27861'
        page_size: int = 10
        offset: int = 0

        def search(self, update_total=True):
            # update_total is True for new searches, but not when paginating
            # reset offset for new searches:
            if update_total:
                self.offset = 0
            params = {"offset": self.offset, "limit": self.page_size}
            if self.instrument_name is not None:
                instrument_id = next(instr['id'] for instr in all_instruments if instr['alias'] == self.instrument_name)
                params['instrument_id'] = instrument_id
            if self.participant_name is not None:
                params['participant_name'] = f"%{self.participant_name}%"
            if self.experiment_title is not None:
                params['title'] = f"%{self.experiment_title}%"
            if self.experiment_id:
                params['id'] = f"{self.experiment_id}"
            if update_total:
                full_count_params = {"full_count": True}
                full_count_params.update(params)
                full_count_result = requests.get(f"{NCNR_METADATA_API_URL}/experiments", params=full_count_params).json()
                if len(full_count_result) > 0:
                    experiment_search_state['pagination']['rowsNumber'] = full_count_result[0].get("full_count", 0)
                    experiment_search_state['pagination']['page'] = 0

            r = requests.get(f"{NCNR_METADATA_API_URL}/experiments", params=params).json()
            experiment_search_state['rows'] = r
            experiments_table.refresh()

        def pagination_request_handler(self, request):
            # request is of type GenericEventArguments
            self.set_page(request.args["pagination"]["page"])

        def set_page(self, new_page: int):
            self.offset = self.page_size * (new_page - 1)
            experiment_search_state['pagination']['page'] = new_page
            self.search(update_total=False)

    experiment_search_params = ExperimentSearchParams()

    @dataclass
    class DatafileSearchParams:
        experiment_id: Optional[str] = None
        filename_substring: str = '' # '68869'
        page_size: int = 10
        offset: int = 0

        def search(self, update_total=True):
            # update_total is True for new searches, but not when paginating
            # reset offset for new searches:
            print('experiment_id: ', self.experiment_id)
            if self.experiment_id is None or self.experiment_id == '':
                self.offset = 0
                datafile_search_state['pagination']['rowsNumber'] = 0
                datafile_search_state['pagination']['page'] = 0
                datafile_search_state['rows'] = []

            else:
                if update_total:
                    self.offset = 0
                params = { "offset": self.offset, "limit": self.page_size, 'experiment_id': self.experiment_id }
                if self.filename_substring:
                    params['filename'] = f"%{self.filename_substring}%"
                if update_total:
                    full_count_params = {"full_count": True}
                    full_count_params.update(params)
                    full_count_result = requests.get(f"{NCNR_METADATA_API_URL}/datafiles", params=full_count_params).json()
                    if len(full_count_result) > 0:
                        datafile_search_state['pagination']['rowsNumber'] = full_count_result[0].get("full_count", 0)
                    datafile_search_state['pagination']['page'] = 0
                r = requests.get(f"{NCNR_METADATA_API_URL}/datafiles", params=params).json()
                datafile_search_state['rows'] = r

            datafiles_table.refresh()

        def pagination_request_handler(self, request):
            # request is of type GenericEventArguments
            self.set_page(request.args["pagination"]["page"])

        def set_page(self, new_page: int):
            self.offset = self.page_size * (new_page - 1)
            datafile_search_state['pagination']['page'] = new_page
            self.search(update_total=False)

    datafile_search_params = DatafileSearchParams()

    @dataclass
    class TimeBinSettings:
        duration: float = 1
        start: float = 0
        end: float = 1
        num_bins: int = 100
        bin_width: float = 1
        use_num: bool = True

        def __setattr__(self, name: str, value: Any) -> None:
            if name == 'duration':
                object.__setattr__(self, 'start', 0.0)
                object.__setattr__(self, 'end', value)
            object.__setattr__(self, name, value)
            self.update()

        def update(self):
            start = self.start
            end = self.end
            duration = self.duration
            if start is None:
                start = 0
            elif start < 0:
                start = duration + start
            if end is None:
                end = duration
            elif end < 0:
                end = duration + end

            if self.use_num and self.num_bins is not None and self.num_bins > 0:
                bin_width = (end - start)/(self.num_bins)
                object.__setattr__(self, 'bin_width', bin_width)
            elif self.bin_width > 0:
                num_bins = np.ceil((end - start) / self.bin_width)
                object.__setattr__(self, 'num_bins', num_bins)

        def get_rebin_time_bins_object(self):
            edges = binning_client._lin_edges(self.duration, self.start, self.end, self.bin_width)
            return binning_models.TimeBins(edges=edges, mask=None, mode='time')
        
        def get_summary_time_bins_object(self):
            # drop any partially-filled bins right after zero, outside of self.start
            # (to align bins with self.start)
            offset_start = self.start % self.bin_width
            edges = binning_client._lin_edges(self.duration, offset_start, self.duration, self.bin_width)
            return binning_models.TimeBins(edges=edges, mask=None, mode='time')
    
    time_bin_settings = TimeBinSettings()

    SESSION_LOOKUP[session_id]['time_bin_settings'] = time_bin_settings

    ui.page_title("CHRNS rebinning")
    ui.add_head_html('''
        <style>
            .sticky-selector-column th:first-child, .sticky-selector-column td:first-child {
                position: sticky;
                left: 0;
                z-index: 1;
                background-color: white;
            }

            .sticky-selector-column tr:first-child th {
                position: sticky;
                top: 0;
                z-index: 1;
                background-color: white;
            }
        </style>
    ''')

    ui.add_head_html('''
    <style>
        #app {
            height: 100%;
            width: 100%;
        }

        main.q-page, div.nicegui-content {
            height: 100%;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            flex: 1 1 0;
        }

        div.plot-container.plotly, div.js-plotly-plot {
            height: 100%;
        }
    </style>
    ''')

    with ui.header(elevated=False).style('background-color: #3874c8').classes('items-center justify-normal'):
        ui.image('/static/chrns-3-smaller.png').classes('w-48')
        ui.label('CHRNS event rebinning').classes ('text-h4 pl-12')
        ui.label('')
        # ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white')


    with ui.tabs().classes('w-full') as tabs:
        exp_selector = ui.tab('Select Experiment')
        file_selector = ui.tab('Select File')
        rebinning_params = ui.tab('Rebinning Params')
        # tabs.on('update:model-value', lambda e: print('rebinning view activated', e))
    with ui.tab_panels(tabs, value=exp_selector).classes('w-full flex-1'):
        with ui.tab_panel(exp_selector):
            with ui.row().classes("w-full"):
                with ui.column():
                    select_instrument = ui.select(
                        available_instrument_names,
                        label="instrument",
                        value=None,
                        clearable=True,
                        on_change=experiment_search_params.search,
                        ).classes('w-32').bind_value(experiment_search_params, 'instrument_name')
                    participant_name = ui.input(
                        label='Participant name',
                        placeholder='start typing',
                        on_change=experiment_search_params.search,
                        ).bind_value(experiment_search_params, 'participant_name').props('clearable')
                    experiment_title = ui.input(
                        label='Experiment title',
                        placeholder='start typing',
                        on_change=experiment_search_params.search,
                        ).bind_value(experiment_search_params, 'experiment_title').props('clearable')
                    experiment_id_input = ui.input(
                        label='Experiment ID',
                        placeholder='start typing',
                        on_change=experiment_search_params.search,
                        ).bind_value(experiment_search_params, 'experiment_id')
                    with ui.row():
                        ui.label('Selected:')
                        ui.label().bind_text_from(experiment_search_state, 'selection')
                with ui.column().style("flex:1"):
                    experiments_table()

        with ui.tab_panel(file_selector):
            with ui.row().classes('w-full justify-between'):
                filename_search = ui.input(
                        label='Filename search',
                        placeholder='start typing',
                        on_change=datafile_search_params.search,
                        ).bind_value(datafile_search_params, 'filename_substring').props('clearable')
                with ui.row():
                    experiment_id = ui.input(
                        label='Experiment ID (IMS)',
                        on_change=datafile_search_params.search,
                    )
                    experiment_id.bind_value_from(experiment_search_state, 'selection')
                    experiment_id.bind_value_to(datafile_search_params, 'experiment_id')

                    filename = ui.input(
                        label='Filename',
                    )
                    filename.bind_value_from(datafile_search_state, 'selection')
                    filename.props("readonly")
                
            datafiles_table()

        with ui.tab_panel(rebinning_params):
            with ui.row():
                experiment_id = ui.input(
                    label='Experiment ID (IMS)',
                )
                experiment_id.bind_value_from(experiment_search_state, 'selection')
                experiment_id.props('readonly')
                experiment_id.bind_value_to(datafile_search_params, 'experiment_id')

                filename = ui.input(
                    label='Filename',
                )
                filename.bind_value_from(datafile_search_state, 'selection')
                filename.props('readonly')

                localdir = ui.input(label='Path').props('readonly')
                localdir.bind_value_from(datafile_search_state, 'selection_path')

                duration_label = ui.input(label='Duration (s)').props('readonly')
                duration_label.bind_value_from(time_bin_settings, 'duration')

                show_summary = ui.button('Show summary', color='positive')
                show_summary.bind_enabled_from(binning_state, 'fetching_summary', backward=lambda v: not v)
                show_summary.bind_visibility_from(binning_state, 'metadata', backward=lambda v: v is not None)
                ui.spinner(size='3em', color='positive').bind_visibility_from(binning_state, 'fetching_summary')

                start_download = ui.link("REBIN+DOWNLOAD", f'/download/nexus/{session_id}').props('target=_blank download=file')
                start_download.classes("bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded no-underline")

                # start_download = ui.button('download rebinned', color='secondary', on_click=lambda: download_binned())
                # start_download.bind_enabled_from(binning_state, 'downloading_nexus', backward=lambda v: not v)
                start_download.bind_visibility_from(binning_state, 'metadata', backward=lambda v: v is not None)
                download_spinner = ui.spinner(size='3em', color='secondary').bind_visibility_from(binning_state, 'downloading_nexus')

            with ui.row():
                # ui.button('load duration', on_click=lambda: get_metadata())
                num_bins = ui.number("Num. bins", format='%d').bind_value(time_bin_settings, 'num_bins')
                num_bins.bind_enabled_from(time_bin_settings, 'use_num')
                use_num = ui.toggle({True: 'Num.', False: 'Width'}).bind_value(time_bin_settings, 'use_num')
                bin_width = ui.number("Bin width").bind_value(time_bin_settings, 'bin_width')
                bin_width.bind_enabled_from(time_bin_settings, 'use_num', backward=lambda v: not v)

                start_time = ui.number("Start").bind_value(time_bin_settings, 'start')
                end_time = ui.number("End").bind_value(time_bin_settings, 'end')

                ui.button('reset (start,end)', on_click=lambda: reset_binning_start_end())

               
            summary_fig = {
                'data': [],
                'layout': dict(
                    title="Time Summary",
                    # margin=dict(l=0, r=0, t=30, b=0),
                    xaxis = dict(title="elapsed time (seconds)", showline=True, mirror=True, showgrid=True),
                    yaxis = dict(title="total counts", showline=True, mirror=True, showgrid=True),
                    template='simple_white',
                ),
                'config': {},
            }

            frame_fig = {
                'data': [],
                'layout': dict(
                    title = "Frame snapshot",
                    xaxis = dict(showline=True, mirror=True, showgrid=True),
                    yaxis = dict(showline=True, mirror=True, showgrid=True, scaleanchor='x', scaleratio=1),
                    template='simple_white',
                ),
                'config': {},
            }

            def handle_click(event):
                # show frame!
                args = event.args
                points = args.pop('points', None)
                if len(points) > 0:
                    point = points[0]
                    data = point.pop('data', {})
                    name = data.pop('name', None)
                    point_number = point['pointNumber']
                    show_frame(name, point_number)


            def handle_box_draw(event):
                # print('relayout: ', event)
                if 'shapes' in event.args:
                    box = event.args['shapes'][-1]
                    start, end = sorted([box['x0'], box['x1']])
                    start = max(start, 0.0)
                    end = min(end, time_bin_settings.duration)
                    time_bin_settings.start, time_bin_settings.end = start, end
                    # print('setting start and end: ', start, end, time_bin_settings.start, time_bin_settings.end)
                    update_binning_start_end()

            with ui.row().classes('w-full flex-1'):
                with ui.column().classes('flex-1 h-full'):
                    with ui.element('div').classes('w-full h-full'):
                        summary_plot = ui.plotly(summary_fig)
                        summary_plot.bind_visibility_from(binning_state, 'fetching_summary', backward=lambda v: not v)
                        summary_plot.on('plotly_relayout', handle_box_draw)
                        summary_plot.on('plotly_click', handle_click)
                with ui.column().classes('flex-1 h-full'):
                    with ui.element('div').classes('w-full h-full'):
                        frame_plot = ui.plotly(frame_fig)

                   
            def show_frame(det_name: str, point_number: int):
                bins = binning_state['bins']
                metadata = binning_state['metadata']
                last_frame = binning_state['last_frame']
                if point_number == last_frame:
                    return
                binning_state['last_frame'] = point_number
                frames = binning_client.get_frames(metadata, bins, start=point_number, stop=point_number+1)
                start_time = bins.edges[point_number]
                end_time = bins.edges[point_number + 1]
                data = frames.data[det_name]
                x = np.arange(data.shape[0])
                y = np.arange(data.shape[1])
                trace = go.Heatmap(x=x, y=y, z=data[:,:,0]).to_plotly_json()
                frame_fig['data'] = [trace]
                frame_fig['layout']['title'] = f'Frame {det_name}: {start_time:.4f} < time < {end_time:.4f} (s)'
                frame_plot.update()

            def get_metadata():
                filename = datafile_search_state.get("selection", None)
                path = datafile_search_state.get("selection_path", None)
                if not filename or not path:
                    ui.notify("file or path is undefined... can't get metadata")
                    binning_state['metadata'] = None
                metadata = binning_client.get_metadata(filename, path=path)
                binning_state['metadata'] = metadata
                time_bin_settings.duration = metadata.duration

            def reset_binning_start_end():
                time_bin_settings.start = 0
                time_bin_settings.end = time_bin_settings.duration
                update_binning_start_end()

            def update_binning_start_end():
                summary_fig['layout']['shapes'] = [
                    {
                        'fillcolor': 'LightBlue',
                        'layer': 'below',
                        'line': {'width': 0},
                        'opacity': 0.2,
                        'type': 'rect',
                        'x0': time_bin_settings.start,
                        'x1': time_bin_settings.end,
                        'xref': 'x',
                        'y0': 0,
                        'y1': 1,
                        'yref': 'y domain'
                    }
                ]
                summary_plot.update()

            async def update_summary():
                metadata = binning_state['metadata']
                if metadata is None:
                    ui.notify("no file loaded")
                    return
                bins = time_bin_settings.get_summary_time_bins_object()
                # bins = binning_client.time_linbins(metadata, interval=interval)
                binning_state['bins'] = bins
                binning_state['fetching_summary'] = True
                # print('starting fetch')
                summary = await run.io_bound(binning_client.get_summary, metadata, bins)
                binning_state['fetching_summary'] = False
                # print('fetching done')
                
                time_bin_edges = summary.bins.edges
                x = time_bin_edges.tolist()
                time_bin_centers = (time_bin_edges[1:] + time_bin_edges[:-1])/2
                total_counts = None
                traces = []
                y_min = float('inf')
                y_max = float('-inf')
                for detname in summary.counts:
                    det_counts = summary.counts[detname]
                    if total_counts is None:
                        total_counts = det_counts.copy()
                    else:
                        total_counts += det_counts
                    y_max = max(y_max, det_counts.max())
                    y_min = min(y_min, det_counts.min())
                    y = det_counts.tolist()
                    # plotting vs edges, need the last value twice:
                    y.append(det_counts[-1])
                    traces.append(go.Scatter(x=x, y=y, name=detname, line_shape='hv').to_plotly_json())

                # y_max = max([trace.y.max() for trace in traces])
                # y_min = min([trace.y.min() for trace in traces])
                y_range = (y_max - y_min)
                display_y_min = y_min - 0.1 * y_range
                display_y_max = y_max + 0.1 * y_range
                x_max = x_range = metadata.duration
                x_min = 0.0
                display_x_min = x_min - 0.1 * x_range
                display_x_max = x_max + 0.1 * x_range

                # traces.append(go.Scatter(x=time_bin_centers, y=total_counts, name="total"))
                summary_fig['data'] = traces
                summary_fig['layout']['yaxis']['range'] = [display_y_min, display_y_max]
                summary_fig['layout']['xaxis']['range'] = [display_x_min, display_x_max]
                summary_fig['config']['modeBarButtonsToAdd'] = ['drawrect']
                update_binning_start_end()

            show_summary.on('click', update_summary)

            # not using?
            async def download_binned():
                metadata: binning_models.MetadataReply = binning_state['metadata']
                if metadata is None:
                    ui.notify("no file loaded")
                    return
                bins = time_bin_settings.get_rebin_time_bins_object()
                binning_state['downloading_nexus'] = True
                nexus_reply = await run.io_bound(binning_client.get_nexus, metadata, bins)
                binning_state['downloading_nexus'] = False
                orig_filename = metadata.measurement.filename
                orig_path = pathlib.Path(orig_filename)
                file_suffixes = ''.join(orig_path.suffixes)
                file_stem = re.sub(f"{file_suffixes}$", '', orig_filename)
                new_filename = f"{file_stem}_rebinned{file_suffixes}"
                nexus_bytes = base64.b64decode(nexus_reply.base64_data)
                ui.download(nexus_bytes, filename=new_filename)
                # def iterfile():
                #     with io.BytesIO(nexus_bytes) as file_like:
                #         yield from file_like
                
                # headers = {
                #     "Content-Disposition": f'attachment; filename="{new_filename}"', "Content-Type": "application/hdf5"}
                # return StreamingResponse(iterfile(), headers=headers)

    await client.connected()
    await client.disconnected()
    SESSION_LOOKUP.pop(session_id)
                
ui.run(favicon=FAVICON)
