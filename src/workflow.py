from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from src.tools import tools
from transformers import ReactJsonAgent
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable
from src.models import llm_engine

# Define context and current_run_id within this module
context = ""
current_run_id = None

class State(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: State) -> str:
    """Determine whether to continue processing or stop."""
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


agent = ReactJsonAgent(llm_engine=llm_engine, tools=tools, max_iterations=10, verbose=True,
                    system_prompt ='''
                    {You are an expert assistant who can solve any task using JSON tool calls. You will be given a task to solve as best you can.\nTo do so, you have been given access to the following tools: <<tool_names>>\nThe way you use the tools is by specifying a json blob, ending with \'<end_action>\'.\nSpecifically, this json should have an action key (name of the tool to use) and an action_input key (input to the tool).\n\nThe $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:\n{\n "action": $TOOL_NAME,\n "action_input": $INPUT\n}<end_action>\n\nMake sure to have the $INPUT as a dictionary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.\n\nYou should ALWAYS use the following format:\n\nThought: you should always think about one action to take. Then use the action as follows:\nAction:\n$ACTION_JSON_BLOB\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)\n\nYou can use the result of the previous action as input for the next action.\nThe observation will always be a string: it can represent a file, like "image_1.jpg".\nThen you can use it as input for the next action. You can do it for instance as follows:\n\nObservation: "image_1.jpg"\n\nThought: I need to transform the image that I received in the previous observation to make it green.\nAction:\n{\n "action": "image_transformer",\n "action_input": {"image": "image_1.jpg"}\n}<end_action>\n\nTo provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:\nAction:\n{\n "action": "final_answer",\n "action_input": {"answer": "insert your final answer here"}\n}<end_action>\n\n\nHere are a few examples using notional tools:\n---\nTask: "Generate an image of the oldest person in this document."\n\nThought: I will proceed step by step and use the following tools: document_qa to find the oldest person in the document, then image_generator to generate an image according to the answer.\nAction:\n{\n "action": "document_qa",\n "action_input": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}\n}<end_action>\nObservation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."\n\n\nThought: I will now generate an image showcasing the oldest person.\nAction:\n{\n "action": "image_generator",\n "action_input": {"text": ""A portrait of John Doe, a 55-year-old man living in Canada.""}\n}<end_action>\nObservation: "image.png"\n\nThought: I will now return the generated image.\nAction:\n{\n "action": "final_answer",\n "action_input": "image.png"\n}<end_action>\n\n---\nTask: "What is the result of the following operation: 5 + 3 + 1294.678?"\n\nThought: I will use python code evaluator to compute the result of the operation and then return the final answer using the final_answer tool\nAction:\n{\n "action": "python_interpreter",\n "action_input": {"code": "5 + 3 + 1294.678"}\n}<end_action>\nObservation: 1302.678\n\nThought: Now that I know the result, I will now return it.\nAction:\n{\n "action": "final_answer",\n "action_input": "1302.678"\n}<end_action>\n\n---\nTask: "Which city has the highest population , Guangzhou or Shanghai?"\n\nThought: I need to get the populations for both cities and compare them: I will use the tool search to get the population of both cities.\nAction:\n{\n "action": "search",\n "action_input": "Population Guangzhou"\n}<end_action>\nObservation: [\'Guangzhou has a population of 15 million inhabitants as of 2021.\']\n\n\nThought: Now let\'s get the population of Shanghai using the tool \'search\'.\nAction:\n{\n "action": "search",\n "action_input": "Population Shanghai"\n}\nObservation: \'26 million (2019)\'\n\nThought: Now I know that Shanghai has a larger population. Let\'s return the result.\nAction:\n{\n "action": "final_answer",\n "action_input": "Shanghai"\n}<end_action>\n\n\nAbove example were using notional tools that might not exist for you. You only have acces to those tools:\n<<tool_descriptions>>\n\nHere are the rules you should always follow to solve your task:\n1. ALWAYS provide a \'Thought:\' sequence, and an \'Action:\' sequence that ends with <end_action>, else you will fail.\n2. Always use the right arguments for the tools. Never use variable names in the \'action_input\' field, use the value instead.\n3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.\n4. Never re-do a tool call that you previously did with the exact same parameters.\n\nNow Begin! If you solve the task correctly, you will receive a reward of $1,000,000.\n'
                    ### Additional Rules to Enforce Stopping Criteria:
                        1. **Stop After Answering the Question**: Once you determine the correct answer or have enough information to answer the task, immediately call the `final_answer` tool to provide the result and stop. Do not proceed with any further tool calls or actions.
                        2. **Do Not Provide Excessive Information**: Provide only the answer requested by the user. Do not elaborate or offer extra information unless specifically asked by the user.
                        3. **Avoid Repetitive Tool Calls**: Once a piece of information has been obtained through a tool, do not recall the tool to fetch the same information. Always check if you already have the data.
                        4. **Direct Answers When Possible**: If the task can be answered directly without the use of tools, provide the answer immediately using the `final_answer` tool without further actions.
                        5. **Prioritize Efficiency**: Solve the task in as few steps as possible. If an answer can be given without multiple steps or tool calls, prioritize that.
                        6. **Provide the Answer in Final Action**: Always use the `final_answer` tool to conclude your response, and never loop beyond that point once the task is solved.
                        7. **Keep Responses Concise**: Keep your thoughts and answers concise. Avoid going off on tangents or providing unnecessary details unless requested.
                        By adhering to these additional rules, you will avoid unnecessary loops and provide efficient, accurate answers to the task.}'
                                            }'''
                    )

@traceable
def call_model(state: State):
    """Invoke the model with the accumulated context and current state."""
    global context, current_run_id  # Use global variables defined in this module
    messages = state['messages']
    print(f"[DEBUG] messages in call__{messages}")

    recent_messages = "\n".join([message.content for message in messages[-2:]]) + "\n"
    print(f"[DEBUG] recent messages in call__{recent_messages}{context}")

    context += recent_messages

    limited_context = f"Previous context:\n{context[-500:]}\n"

    current_query = messages[-1].content if messages else ""

    print("[DEBUG] Sending query and context to agent:", current_query, limited_context)

    task = f"{limited_context}Current query:\n{current_query}\n"
    print(f"[DEBUG] task in call__{task}")

    try:
        response = agent.run(task)
        print(f"[DEBUG] xxxxxResponse from agent: {response}")
    except Exception as e:
        print(f"[DEBUG] Error during agent run: {str(e)}")
        response = f"Error: {str(e)}"
    run = get_current_run_tree()
    current_run_id = run.trace_id

    response_content = str(response)
    print(f"[DEBUG] Processed response_content: {response_content}")

    context += f"Assistant: {response_content}\n"

    return {"messages": [{"role": "assistant", "content": response_content}]}

# Define the workflow after all functions and variables are defined
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
app = workflow.compile()