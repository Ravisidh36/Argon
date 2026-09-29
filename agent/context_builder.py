class ContextBuilder:

    def build(self, question, context):
        return (
            f"You are a helpful assistant. Answer the question using ONLY the information below.\n"
            f"Do NOT use any outside knowledge. If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )