from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import pandas as pd

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # We can pass df into state if needed, but since we recreate tools with df, we don't strictly need it in state for the graph
    # However, keeping requested_charts in state is useful
    requested_charts: List[str]
