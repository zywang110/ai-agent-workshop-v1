import openai
import json
import base64
from dotenv import load_dotenv

from image_embedding import image_embedding_store
from details.generate_tools_schema import generate_json_schema

load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI()

dataset_dir = 'dataset'
img_store = image_embedding_store(dataset_dir)

def find_animals_in_an_image(path: str):
    """
    Get the animal category of the image

    @param path: file path to the image
    """
    with open(f"{path}", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Describe all the animal types and quantities in the image. Only use one word in lowercase for the animal type. Return format example: {'cat': 2, 'dog': 1}"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are the animals in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
    )
    print(f"In the image {path}\n {response.choices[0].message.content}")
    return response.choices[0].message.content
    
def get_all_files():
    """
    Finds all the files to process
    
    @returns: a list of file paths
    """
    return img_store.get_all_files()

# generate_json_schema is a convenient helper function to generate a JSON schema for the functions, so we can skip the tedious work of writing the schema manually.
functions = [generate_json_schema(f) for f in [
    find_animals_in_an_image, 
    get_all_files
    ]]

def run_image_agent(query: str):

    assistant = client.beta.assistants.create(
        model='gpt-4o-2024-08-06',
        instructions="You are an image assistant. Your job is helping the user identify and understand the images",
        tools=functions,
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
    query_animals_feet = "How many feet are there in total for all the animals in the images in the dataset folder"
    query_animals_feet_inverse = "Find a set of images that the total animal feet in those images are equal to 38"

    result = run_image_agent(query_animals_feet)
    print(f"Response from LLM: {result}")

