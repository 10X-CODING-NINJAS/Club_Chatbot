# Chatbot Theme & Testing

## Objective

The repository is being converted from an event chatbot into a permanent **Coding Ninjas club chatbot**.

Your team's responsibility is limited to:

1. **Change the chatbot's system prompt/theme**
2. **Test the chatbot**

Do not modify the vector database, ingestion pipeline, deployment, or retrieval system.

Repository:

`https://github.com/10X-CODING-NINJAS/Club_Chatbot`

---

# 1. Change the System Prompt

The main application file is:

```text
app.py
```

Find the main:

```python
system_prompt = """
...
"""
```

This is the prompt responsible for the chatbot's:

- Identity
- Personality
- Tone
- Response style
- Behavior

The current prompt has the old event/Spider-Bot theme.

Replace it with a **new theme appropriate for the Coding Ninjas club**.

You should decide the theme yourself.

The new theme should:

- Fit a coding/technical student club.
- Be suitable for a permanent club chatbot.
- Have a consistent personality.
- Make responses engaging without becoming excessive.
- Never interfere with factual answers.

---

# 2. Preserve the Important Behavior

While changing the theme, preserve the existing RAG behavior.

The chatbot should still:

- Answer using the retrieved club information.
- Avoid making up facts.
- Say when information is unavailable.
- Give useful and understandable answers.
- Maintain the new personality consistently.

The theme should change **how the chatbot responds**, not what information it retrieves.

---

# 3. Do NOT Modify

Do not modify:

```text
ingest_cn10x.py
2_club_retrieval_pipeline.py
```

Do not modify:

- Qdrant
- Embeddings
- Vector database
- Retrieval logic
- Knowledge files
- Deployment
- Frontend
- LLM configuration

Your task is only the **system prompt + testing**.

---

# 4. Important: Hardcoded Greetings

Some greetings may be handled directly by `app.py` rather than the system prompt.

Therefore, changing the system prompt may not change responses to:

```text
hi
hello
hey
good morning
```

Do not change these separately unless explicitly required.

---

# 5. Testing

After changing the system prompt, run the chatbot and test a small set of questions.

Test:

### Normal club questions

```text
What is the club?
What does the club do?
How can I join?
Who leads the club?
```

### Follow-up

Ask a question and then ask a related follow-up to verify conversation context still works.

### Unknown information

Ask something that is not present in the knowledge base.

Make sure the chatbot does not invent an answer.

### Personality

Ask several different questions and verify that the new theme/personality remains consistent without overwhelming the actual answer.

---

# 6. Final Checklist

```text
[ ] System prompt updated
[ ] New club-appropriate theme implemented
[ ] Old event theme removed
[ ] RAG behavior preserved
[ ] Normal club questions tested
[ ] Follow-up question tested
[ ] Unknown question tested
[ ] Personality consistency tested
[ ] No vector/retrieval/deployment files changed
```

---

# Instructions for an LLM

If an AI coding agent is performing this task:

> Work on `https://github.com/10X-CODING-NINJAS/Club_Chatbot`.
>
> Your task is ONLY to:
>
> 1. Replace the existing chatbot system prompt with a new theme/personality appropriate for the Coding Ninjas club.
> 2. Test the chatbot after the change.
>
> Inspect `app.py` and identify the main `system_prompt`.
>
> Create a coherent club-appropriate personality yourself. Do not ask for a specific theme unless necessary.
>
> Preserve the existing RAG behavior and factuality requirements. The chatbot must continue using retrieved information and must not invent facts.
>
> Do not modify the vector database, ingestion pipeline, retrieval pipeline, embeddings, Qdrant, deployment, frontend, or LLM configuration.
>
> Be aware that some greeting responses may be hardcoded separately from the system prompt.
>
> After making the change, test:
>
> - A few normal club questions
> - A follow-up question
> - An unknown question
> - Several questions to check personality consistency
>
> Confirm that the new theme works while factual/RAG behavior remains intact.
>
> At the end, report:
>
> - Files changed
> - What was changed
> - Tests performed
> - Whether everything works correctly