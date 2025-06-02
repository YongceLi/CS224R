from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.hiql_r import HIQLRAgent
from agents.hiql_reverse import HIQLReverseAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.hiql_reverse_v2 import HIQLReverseV2Agent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    hiqlr=HIQLRAgent,
    hiql_reverse=HIQLReverseAgent,
    hiql_reverse_v2=HIQLReverseV2Agent,
)
