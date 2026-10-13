from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
import pandas as pd
import os

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from tools.patient_tool import get_patient_tools
from tools.analytics_tool import get_analytics_tools
from tools.visualization_tool import get_visualization_tools
from tools.rag_tool import query_medical_knowledge
from tools.specialist_tool import recommend_specialist

def create_agent_graph(df: pd.DataFrame):
    # Initialize tools with current dataframe
    patient_tools = get_patient_tools(df)
    analytics_tools = get_analytics_tools(df)
    viz_tools = get_visualization_tools()
    
    tools = patient_tools + analytics_tools + viz_tools + [query_medical_knowledge, recommend_specialist]
    
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if groq_key:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=groq_key)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_key)
        
    llm_with_tools = llm.bind_tools(tools)
    
    # Define the nodes
    def agent_node(state: AgentState):
        messages = state['messages']
        
        # Ensure system prompt is the first message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
        response = llm_with_tools.invoke(messages)
        
        # Parse for chart requests (we intercept CHART_REQUESTED returned by the tool)
        # But wait, if the tool returned it, it's in a ToolMessage. The agent will then generate a final AIMessage.
        # We can extract requested charts by looking at the AIMessage tool calls if they were made, 
        # or by looking at the ToolMessage contents in the state.
        
        return {"messages": [response]}
        
    def should_continue(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        
        if not last_message.tool_calls:
            return "end"
        return "continue"
        
    def parse_charts(state: AgentState):
        """Node to extract requested charts from tool messages."""
        messages = state['messages']
        requested_charts = state.get('requested_charts', [])
        
        # Look for CHART_REQUESTED in recent tool messages
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if msg.content.startswith("CHART_REQUESTED:"):
                    chart_type = msg.content.split(":")[1]
                    if chart_type not in requested_charts:
                        requested_charts.append(chart_type)
                        
        return {"requested_charts": requested_charts}
        return {"requested_charts": requested_charts}

    def safety_check_node(state: AgentState):
        """Emergency Safety Layer: Intercepts responses with critical vitals."""
        messages = state['messages']
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
            content = last_message.content.lower()
            # Simple keyword detection for critical vitals
            is_critical = False
            if "glucose" in content and any(val in content for val in ["> 250", "> 300", "extremely high", "critical"]):
                is_critical = True
            elif "blood pressure" in content and any(val in content for val in ["> 180", "180/120", "hypertensive crisis"]):
                is_critical = True
                
            if is_critical:
                warning = "\n\n🚨 **EMERGENCY WARNING:** The vitals discussed indicate a critical medical emergency. Please seek immediate medical attention or call emergency services."
                if warning not in last_message.content:
                    # Create a new AIMessage with the appended warning
                    new_msg = AIMessage(content=last_message.content + warning)
                    messages[-1] = new_msg
                    return {"messages": messages}
        return {"messages": messages}

    tool_node = ToolNode(tools)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_node)
    workflow.add_node("parse_charts", parse_charts)
    workflow.add_node("safety_check", safety_check_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": "parse_charts"
        }
    )
    # After tool execution, go back to agent
    workflow.add_edge("action", "agent")
    # After parse_charts, run safety check
    workflow.add_edge("parse_charts", "safety_check")
    # After safety check, we are done
    workflow.add_edge("safety_check", END)
    
    return workflow.compile()
