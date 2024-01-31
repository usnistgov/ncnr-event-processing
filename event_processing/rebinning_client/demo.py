from dataclasses import dataclass, asdict
import json
from typing import Optional
from nicegui import ui
import requests

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
API_ROOT = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1"

experiment_data = {
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

datafiles_data = {
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

@ui.refreshable
def experiments_table():
    experiment_columns = [
        {'name': 'experiment_id', 'label': 'Experiment ID', 'field': 'id', 'required': True, 'align': 'left', 'style': 'width: 8em;'},
        {'name': 'title', 'label': 'Title', 'field': 'title', 'align': 'left', 'style': 'max-width:2px;'},
        {'name': 'participants', 'label': 'Participants', 'field': 'participant_names', 'align': 'left', ':format': 'value => JSON.parse(value).join(", ")', 'style': 'max-width:2px;'},
    ]
    def select_handler(e):
        selected_id = e.selection[0]['id'] if len(e.selection) > 0 else ''
        experiment_data['selection'] = selected_id
        if selected_id != '':
            ui.tab_panels(tabs).value = file_selector

    table = ui.table(
        columns=experiment_columns,
        rows=experiment_data['rows'],
        row_key='id',
        pagination=experiment_data['pagination'],
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
        datafiles_data['selection'] = selected_id
        datafiles_data['selection_path'] = selected_path
        if selected_id != '':
            ui.tab_panels(tabs).value = rebinning_params
    
    extra_cols = get_metadata_columns(datafiles_data['rows'])

    table = ui.table(
        columns=columns + extra_cols,
        rows=datafiles_data['rows'],
        row_key='filename',
        pagination=datafiles_data['pagination'],
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
    experiment_id: Optional[str] = None
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
            full_count_result = requests.get(f"{API_ROOT}/experiments", params=full_count_params).json()
            if len(full_count_result) > 0:
                experiment_data['pagination']['rowsNumber'] = full_count_result[0].get("full_count", 0)
                experiment_data['pagination']['page'] = 0

        r = requests.get(f"{API_ROOT}/experiments", params=params).json()
        experiment_data['rows'] = r
        experiments_table.refresh()

    def pagination_request_handler(self, request):
        # request is of type GenericEventArguments
        self.set_page(request.args["pagination"]["page"])

    def set_page(self, new_page: int):
        self.offset = self.page_size * (new_page - 1)
        experiment_data['pagination']['page'] = new_page
        self.search(update_total=False)

experiment_search_params = ExperimentSearchParams()

@dataclass
class DatafileSearchParams:
    experiment_id: Optional[str] = None
    filename_substring: str = ''
    page_size: int = 10
    offset: int = 0

    def search(self, update_total=True):
        # update_total is True for new searches, but not when paginating
        # reset offset for new searches:
        print('experiment_id: ', self.experiment_id)
        if self.experiment_id is None or self.experiment_id == '':
            self.offset = 0
            datafiles_data['pagination']['rowsNumber'] = 0
            datafiles_data['pagination']['page'] = 0
            datafiles_data['rows'] = []

        else:
            if update_total:
                self.offset = 0
            params = { "offset": self.offset, "limit": self.page_size, 'experiment_id': self.experiment_id }
            if self.filename_substring:
                params['filename'] = f"%{self.filename_substring}%"
            if update_total:
                full_count_params = {"full_count": True}
                full_count_params.update(params)
                full_count_result = requests.get(f"{API_ROOT}/datafiles", params=full_count_params).json()
                if len(full_count_result) > 0:
                    datafiles_data['pagination']['rowsNumber'] = full_count_result[0].get("full_count", 0)
                datafiles_data['pagination']['page'] = 0
            r = requests.get(f"{API_ROOT}/datafiles", params=params).json()
            datafiles_data['rows'] = r

        datafiles_table.refresh()

    def pagination_request_handler(self, request):
        # request is of type GenericEventArguments
        self.set_page(request.args["pagination"]["page"])

    def set_page(self, new_page: int):
        self.offset = self.page_size * (new_page - 1)
        datafiles_data['pagination']['page'] = new_page
        self.search(update_total=False)

datafile_search_params = DatafileSearchParams()

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

with ui.header(elevated=False).style('background-color: #3874c8').classes('items-center justify-between'):
    ui.image('https://ncnr.nist.gov/ncnrdata/metadata/chrns-3-smaller.png').classes('w-48')
    ui.label('CHRNS event rebinning')
    ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('flat color=white')


with ui.tabs().classes('w-full') as tabs:
    exp_selector = ui.tab('Select Experiment')
    file_selector = ui.tab('Select File')
    rebinning_params = ui.tab('Rebinning Params')
with ui.tab_panels(tabs, value=exp_selector).classes('w-full'):
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
                    ui.label().bind_text_from(experiment_data, 'selection')
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
                experiment_id.bind_value_from(experiment_data, 'selection')
                experiment_id.bind_value_to(datafile_search_params, 'experiment_id')

                filename = ui.input(
                    label='Filename',
                )
                filename.bind_value_from(datafiles_data, 'selection')
                filename.props("readonly")
               
        datafiles_table()

    with ui.tab_panel(rebinning_params):
        with ui.row():
            experiment_id = ui.input(
                label='Experiment ID (IMS)',
            )
            experiment_id.bind_value_from(experiment_data, 'selection')
            experiment_id.props('readonly')
            experiment_id.bind_value_to(datafile_search_params, 'experiment_id')

            filename = ui.input(
                label='Filename',
            )
            filename.bind_value_from(datafiles_data, 'selection')
            filename.props('readonly')

            localdir = ui.input(label='Path').props('readonly')
            localdir.bind_value_from(datafiles_data, 'selection_path')


ui.run(favicon=FAVICON)
