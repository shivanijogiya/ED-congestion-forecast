"""
Static hospital graph topology.
Nodes = departments, Edges = relationships (transfer, shared_resource, proximity).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Department:
    dept_id: str
    dept_name: str
    dept_type: str          # triage | resus | fast_track | obs | radiology | boarding | discharge | waiting
    capacity: int
    floor_number: int
    lat: float
    lon: float

@dataclass
class Edge:
    source: str
    target: str
    edge_type: str          # TRANSFER | SHARED_RESOURCE | PROXIMITY
    base_weight: float      # static prior weight [0, 1]

@dataclass
class Hospital:
    hospital_id: str
    hospital_name: str
    departments: List[Department]
    edges: List[Edge]

# ─── Hospital definitions ─────────────────────────────────────────────────────

def _build_hospital(hid: str, name: str, lat: float, lon: float) -> Hospital:
    """Constructs a realistic 7-department ED graph for a given hospital."""
    prefix = hid

    depts = [
        Department(f"{prefix}_TRIAGE",    "Triage",             "triage",      12,  1, lat + 0.001, lon),
        Department(f"{prefix}_RESUS",     "Resuscitation",      "resus",        4,  1, lat,         lon + 0.001),
        Department(f"{prefix}_FASTTRACK", "Fast Track",         "fast_track",  20,  1, lat - 0.001, lon),
        Department(f"{prefix}_OBS",       "Observation",        "obs",         16,  2, lat,         lon - 0.001),
        Department(f"{prefix}_RADIOLOGY", "Radiology",          "radiology",    8,  2, lat + 0.002, lon + 0.001),
        Department(f"{prefix}_BOARDING",  "Inpatient Boarding", "boarding",    12,  1, lat - 0.001, lon + 0.001),
        Department(f"{prefix}_DISCHARGE", "Discharge Lounge",   "discharge",   10,  1, lat + 0.001, lon - 0.001),
    ]

    edges = [
        # TRANSFER edges — high acuity flow
        Edge(f"{prefix}_TRIAGE",    f"{prefix}_RESUS",     "TRANSFER",        0.8),
        Edge(f"{prefix}_TRIAGE",    f"{prefix}_FASTTRACK", "TRANSFER",        0.6),
        Edge(f"{prefix}_TRIAGE",    f"{prefix}_OBS",       "TRANSFER",        0.4),
        Edge(f"{prefix}_RESUS",     f"{prefix}_OBS",       "TRANSFER",        0.7),
        Edge(f"{prefix}_RESUS",     f"{prefix}_BOARDING",  "TRANSFER",        0.5),
        Edge(f"{prefix}_OBS",       f"{prefix}_BOARDING",  "TRANSFER",        0.6),
        Edge(f"{prefix}_OBS",       f"{prefix}_DISCHARGE", "TRANSFER",        0.4),
        Edge(f"{prefix}_FASTTRACK", f"{prefix}_DISCHARGE", "TRANSFER",        0.7),
        Edge(f"{prefix}_BOARDING",  f"{prefix}_DISCHARGE", "TRANSFER",        0.3),

        # SHARED_RESOURCE edges — staff/equipment sharing
        Edge(f"{prefix}_TRIAGE",    f"{prefix}_RESUS",     "SHARED_RESOURCE", 1.0),
        Edge(f"{prefix}_OBS",       f"{prefix}_RADIOLOGY", "SHARED_RESOURCE", 0.9),
        Edge(f"{prefix}_FASTTRACK", f"{prefix}_RADIOLOGY", "SHARED_RESOURCE", 0.7),

        # PROXIMITY edges — physical adjacency
        Edge(f"{prefix}_TRIAGE",    f"{prefix}_FASTTRACK", "PROXIMITY",       0.5),
        Edge(f"{prefix}_RESUS",     f"{prefix}_OBS",       "PROXIMITY",       0.6),
        Edge(f"{prefix}_OBS",       f"{prefix}_DISCHARGE", "PROXIMITY",       0.4),
    ]

    return Hospital(hospital_id=hid, hospital_name=name, departments=depts, edges=edges)


HOSPITALS: List[Hospital] = [
    _build_hospital("H1", "City General Hospital",    23.0225, 72.5714),
    _build_hospital("H2", "North Regional Medical",  23.0800, 72.5200),
    _build_hospital("H3", "Sunrise Health Centre",   22.9900, 72.6000),
    _build_hospital("H4", "Metro Emergency Hospital",23.0500, 72.5500),
    _build_hospital("H5", "Westside Medical",        23.0100, 72.5100),
    _build_hospital("H6", "Eastview Hospital",       23.0600, 72.6200),
]


def get_hospital_map() -> Dict[str, Hospital]:
    return {h.hospital_id: h for h in HOSPITALS}


def get_all_dept_ids() -> List[str]:
    return [d.dept_id for h in HOSPITALS for d in h.departments]


def get_dept_capacity(hospital_id: str, dept_id: str) -> int:
    h_map = get_hospital_map()
    hospital = h_map.get(hospital_id)
    if not hospital:
        return 10
    for d in hospital.departments:
        if d.dept_id == dept_id:
            return d.capacity
    return 10
