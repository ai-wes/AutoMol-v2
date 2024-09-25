import json
import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, OllamaClient
from knowledge_storm.rm import YouRM

# API key setup (for retrieval model)
youcom_api_key = os.getenv('YDC_API_KEY')

# Language model configuration for locally hosted Ollama model
lm_configs = STORMWikiLMConfigs()

# Define the locally hosted model using Ollama API
ollama_url = "http://localhost:11434"
ollama_api_key = "ollama"  # This is required but not used by Ollama, it just needs a placeholder

# Use OllamaClient to connect to the local model
gpt_4o_mini_local = OllamaClient(model='llama2', host=ollama_url, api_key=ollama_api_key)
gpt_4o_local = OllamaClient(model='llama2', host=ollama_url, api_key=ollama_api_key)

# Assign models to tasks
lm_configs.set_conv_simulator_lm(gpt_4o_mini_local)
lm_configs.set_question_asker_lm(gpt_4o_mini_local)
lm_configs.set_outline_gen_lm(gpt_4o_local)
lm_configs.set_article_gen_lm(gpt_4o_local)
lm_configs.set_article_polish_lm(gpt_4o_local)

# Runner arguments for the STORM engine
engine_args = STORMWikiRunnerArguments(search_top_k=5, max_conv_turn=5, max_perspective=3)

# Retrieval setup (e.g., You.com search engine)
rm = YouRM(ydc_api_key=youcom_api_key, k=engine_args.search_top_k)

# Create the STORM runner
runner = STORMWikiRunner(engine_args, lm_configs, rm)

# Define the sections of the article
sections = {
    "Abstract": "AutoMol-v2 is a novel autonomous system for de novo compound design...",
    "Introduction": "A. Background on de novo molecular design...",
    "System Architecture": "A. Overall pipeline description...",
    "Results and Discussion": "A. Performance of natural language instruction processing...",
    "Conclusion": "AutoMol-v2 represents a significant advancement in autonomous molecular design...",
}

# Iteratively run the STORM system on each section
for section_title, section_content in sections.items():
    print(f"Processing section: {section_title}")
    
    # Run STORM on the section title and content iteratively
    runner.run(
        topic=section_title,              # Use the section title as the topic
        do_research=True,                 # Research the topic (section)
        do_generate_outline=True,         # Generate an outline for the section
        do_generate_article=True,         # Generate the section article based on the outline
        do_polish_article=True            # Polish the section
    )
    
    # Save each section separately
    output_dir = f"./output/automol_v2_{section_title}/"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/{section_title}_article.txt", "w") as f:
        f.write(runner.article)
    
    with open(f"{output_dir}/references.json", "w") as f:
        json.dump(runner.references, f)

    runner.post_run()
    runner.summary()

print("All sections processed!")
