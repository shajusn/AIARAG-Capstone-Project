from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",
)

context = ["This is a test document. " * 100]
query = "What is this document?"

results = compressor.compress_prompt_llmlingua2(context=context, target_token=50)

print(results)
