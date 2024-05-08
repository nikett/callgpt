from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper

from typing import List 


class AbstractTakeawayForClaimTask(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, claim: str, abstract: str) -> str:
        if not claim or not abstract:
            return ""

        question_prefix_template = \
            f"""
Claim: {claim}

Abstract: {abstract}

Now, answer these two questions:
Q1. Is the claim and abstract related or unrelated?
Q2. How can someone accurately extract the main point of the abstract in relation to the claim?(Only extract detail about the salient relation. Do NOT provide any stance about the claim. )
"""
        query = f"""{self.question_prefix}{question_prefix_template.format(claim=claim, abstract=abstract)}"""
        query = f"{query}{self.intra_example_sep}"
        return query
    
    def make_chat_query(self, claim: str, abstract: str) -> List:
        if not claim or not abstract:
            return ""
        # The chat prompt is the same as the query prompt, but with the role of the user (or agent) specified.
        messages=[
                {
                    "role": "system",
                    "content": "You are an expert scientist who understand claims made in research papers.",
                },
                {
                    "role": "user",
                    "content": "Tell me fun things to do in San Francisco. Separate answers by \n\n.",
                }
            ]
        
        return messages

    def __call__(self, claim: str, abstract: str, is_chat=False) -> str:
        generation_query = self.make_query(claim=claim, abstract=abstract) if not is_chat else self.make_chat_query(claim=claim, abstract=abstract)

        generated_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=500,
            stop_token="###",
            temperature=0.0,
        )

        # (extract answers) A1.xxx\n\nA2.xxx
        generated_sent = generated_sent.strip()  # gpt3 turbo adds newline in the beginning so strip it.
        generated_answers = generated_sent.split("\n\n")
        if len(generated_answers) != 2:
            # second attempt
            generated_answers = generated_sent.split("\n")

        # first relevant_sent is just "A2. " so ignore it.
        relation = ""
        takeaway_sent = ""
        try:
            relation=generated_answers[0].strip()
            takeaway_sent=generated_answers[1].strip()
            # Make the abstract takeaways txt cleaner. (remove: Q2. The revised claim could be: )
            # {'A0': 'Q2. The revised claim could be: "Delayed diagnosis of cervical cancer is a major contributor to increasing rates of cervical cancer in Ethiopia."', 'A1': 'Q2. The claim can be rewritten to: Cervical cancer rates have increased in Ethiopia since the launch of the Gynecologic Oncology Fellowship Training Program at St. Paul’s Hospital Millennium Medical college in 2016.', 'A2': 'Q2. The claim can be rewritten to: "Cervical cancer screening practice among age-eligible women in Wolaita Zone hospitals in Southern Ethiopia is low, despite age, being an adherence supporter, source of information from health care professionals, history of multiple sexual partners, sexually transmitted infection, knowledge and attitude being important predictors of cervical cancer screening practice."', 'A3': 'Q2. The revised claim could be: "Cervical cancer screening and treatment services in South West Shoa Zone of Oromia Region, Ethiopia, have revealed an increasing rate of cervical cancer cases."', 'A4': 'Q2. The claim can be rewritten to: "Cervical cancer screening practices and associated factors among females of reproductive age in Durame, Southern Ethiopia are increasing."', 'A5': 'Q2. The rewritten claim could be: "The utilization of cervical cancer screening services and its predictors among eligible women in Ethiopia are being assessed in a systematic review and meta-analysis."'}
            takeaway_sent = " ".join(takeaway_sent.split(":" if ":" in takeaway_sent else ".")[1:])
        except Exception as exc:
            print(f"Exception caught in extracting rel or sents in claim abstract link: {exc}.\n"
                  f"Could not extract from generated text: {generated_sent}")

        return relation, takeaway_sent


if __name__ == '__main__':
    # Load from the cached file. This should run without a key.
    openai_wrapper = OpenAIWrapper(cache_path="cache.jsonl")
    gpt4turbo = AbstractTakeawayForClaimTask(engine="gpt-4-turbo", openai_wrapper=openai_wrapper)
    gpt35turbo = AbstractTakeawayForClaimTask(engine="gpt-3.5-turbo", openai_wrapper=openai_wrapper)
    
    sample_claim = "snow makes people sick."
    sample_claim2 = "snow does not make people sick."
    sample_claim3 = "snow cannot make people sick."
    sample_abstract = "It would occupy a long time to give an account of the progress of cholera over different parts of the world, with the devastation it has caused in some places, whilst it has passed lightly over others, or left them untouched; and unless this account could be accompanied with a description of the physical condition of the places, and the habits of the people, which I am unable to give, it would be of little use. There are certain circumstances, however, connected with the progress of cholera, which may be stated in a general way. It travels along the great tracks of human intercourse, never going faster than people travel, and generally much more slowly. In extending to a fresh island or continent, it always appears first at a sea-port. It never attacks the crews of ships going from a country free from cholera to one where the disease is prevailing, till they have entered a port, or had intercourse with the shore. Its exact progress from town to town cannot always be traced; but it has never appeared except where there has been ample opportunity for it to be conveyed by human intercourse. There are also innumerable instances which prove the communication of cholera, by individual cases of the disease, in the most convincing manner. Instances such as the following seem free from every source of fallacy. I called lately to inquire respecting the death of Mrs. Gore, the wife of a labourer, from cholera, at New Leigham Road, Streatham. I found that a son of deceased had been living and working at Chelsea. He came home ill with a bowel complaint, of which he died in a day or two. His death took place on August 18th. His mother, who attended on him, was taken ill on the next day, and died the day following (August 20th). There were no other deaths from cholera registered in any of the metropolitan districts, down to the 26th August, within two or three miles of the above place; the nearest being."
    
    print(f"claim: {sample_claim}\nabstract: {sample_abstract}\n")
    print(f"Engine: gpt-4-turbo")
    print(gpt4turbo(claim=sample_claim, abstract=sample_abstract))
    print(f"Engine: gpt-3.5-turbo")
    print(gpt35turbo(claim=sample_claim, abstract=sample_abstract))

    print(f"claim2: {sample_claim2}\nabstract: {sample_abstract}\n")
    print(f"Engine: gpt-4-turbo")
    print(gpt4turbo(claim=sample_claim2, abstract=sample_abstract))
    print(f"Engine: gpt-3.5-turbo")
    print(gpt35turbo(claim=sample_claim2, abstract=sample_abstract))
    
    print(f"claim3: {sample_claim3}\n")
    print(f"Engine: gpt-4-turbo")
    print(gpt4turbo(claim=sample_claim3, abstract=sample_abstract, is_chat=True))
    print(f"Engine: gpt-3.5-turbo")
    print(gpt35turbo(claim=sample_claim3, abstract=sample_abstract, is_chat=True))

