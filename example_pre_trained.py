from transformers import pipeline, set_seed

# generator = pipeline('text-generation', model="facebook/opt-125m", do_sample=True)
generator = pipeline('text-generation', model='gpt2')
set_seed(32)

N_GENERATE = 30
lyrics = input("Enter the lyrics: ")
while lyrics != "exit":
    lyrics = "COMPLETE THE LYRICS: " + lyrics
    output = generator(lyrics, max_length=N_GENERATE)
    print(output[0]['generated_text'].split("COMPLETE THE LYRICS: ")[1])
    lyrics = input("Enter the lyrics: ")