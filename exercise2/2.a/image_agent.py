import openai
import json
import base64
from dotenv import load_dotenv
from PIL import Image

from image_embedding import image_embedding_store

load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI()

dataset_dir = 'dataset'
img_store = image_embedding_store(dataset_dir)

def show_image_from_path(path: str):
    """
    Shows an image from a given path

    @param path: image path
    """
    img = Image.open(path)
    img.show()

show_image_from_path_schema = {
    "type": "function",
    "function": {
        "parameters": {
            "title": "show_image_from_path",
            "type": "object",
            "properties": { 
                "path": {
                    "title": "Path",
                    "description": "image path",
                    "type": "string"
                }
            },
            "required": ["path"]
        },  
        "name": "show_image_from_path",
        "description": "Shows an image from a given path"
    }
}

#TODO: write function that generate image path based on description
def find_image_path_based_on_description():
    pass

find_image_path_based_on_description_schema = {
   # TODO
}

tools = [find_image_path_based_on_description_schema, show_image_from_path_schema]

def run_image_agent(query: str):

    assistant = client.beta.assistants.create(
        model='gpt-4o-2024-08-06',
        instructions="You are an image assistant. Your job is helping the user identify and understand the images",
        tools=tools,
        name="image-agent",
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=query)
    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id,)

    max_turns = 50
    for _ in range(max_turns):
        messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    run_id=run.id,
                    order="desc",
                    limit=1,
                )
        
        if run.status == "completed":
            return  next((
                    content.text.value
                    for content in messages.data[0].content
                    if content.type == "text"
                ),
                None,
            )

        elif run.status == "requires_action":
            func_tool_outputs = []
            for tool in run.required_action.submit_tool_outputs.tool_calls:

                if tool.function.name in globals() and callable(globals()[tool.function.name]):
                    print(f"Calling Function: {tool.function.name}")
                    if tool.function.name == "show_image_from_path":
                        func_output = show_image_from_path(json.loads(tool.function.arguments)["path"])
                    else:
                        func_output = globals()[tool.function.name](**json.loads(tool.function.arguments))
                    func_tool_outputs.append({"tool_call_id": tool.id, "output": str(func_output)})
                else:
                    raise Exception("Function not available")

            # Submit the function call outputs back to OpenAI
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(thread_id=thread.id, run_id=run.id, tool_outputs=func_tool_outputs)

        elif run.status == "failed":
                print(f"Agent run failed for the reason: {run.last_error}")
                break
        else:
            print(f"Run status {run.status} not yet handled")
    else:
        print("Reached maximum reasoning turns, maybe increase the limit?")

if __name__ == "__main__":
    # Testing prompts
    query_show = "Show the image of a cat reading a book"
    query_count_cat = "Count the number images that are cats in the dataset folder"

    result = run_image_agent(query_show)
    print(f"Response from LLM: {result}")

