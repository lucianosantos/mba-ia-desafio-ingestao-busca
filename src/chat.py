import sys
from search import search_prompt

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Pergunta:\n{question}")
    else:
        question = None

    if not question:
        print("Faça sua pergunta:")
        question = input()

    sys.stdout.write("\rCarregando resposta...") 
    sys.stdout.flush()
    result = search_prompt(question)

    sys.stdout.write("\r" + " " * 30 + "\r") 
    if result:
        print("Resposta:")
        print(result)
    else:
        print("Nenhuma pergunta fornecida. Por favor, tente novamente.")

if __name__ == "__main__":
    main()