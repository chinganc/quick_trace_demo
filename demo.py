from opto import trace
from opto.utils.llm import AutoGenLLM
from opto.optimizers import OptoPrime
from opto.optimizers.utils import print_color



llm = AutoGenLLM()  # This uses autogen's OpenAIWrapper based on config provided in OAI_CONFIG_LIST


@trace.bundle()
def query_llm(system_prompt: str, user_prompt: str):
    """ Query the language model of system_prompt with input_prompt."""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    response = llm(
        messages=messages,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def critic(response):
    system_prompt = "Check if the response is sunny. If not, give hints that the weather is sunny."
    return query_llm(system_prompt, response).data  # we just reuse the previous code.



# Start the optimization loop

prompt = trace.node("You're a helpful assistant", trainable=True)  # init param
user_prompt = "What is the weather like today?"
optimizer = OptoPrime([prompt])

for i in range(5):
    response = query_llm(prompt, user_prompt)
    feedback = critic(response)
    print_color(f"Prompt: {prompt.data}\n", 'blue')
    print_color(f"Response: {response.data}\n", 'red')
    print_color(f"Feedback: {feedback}\n", 'green')
    optimizer.zero_feedback()
    optimizer.backward(response, feedback)
    optimizer.step(verbose='output')
