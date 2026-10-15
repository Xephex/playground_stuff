# n8n workflows for the LLM kit

Two ready-made workflows that talk to the model served by the kit:

- `llm-kit-chat-agent.json`: a chat window with an AI Agent (model `granite-4.2-8b`).
- `llm-kit-summarize-webhook.json`: send text to a web address, get a 3-bullet summary back (model `qwen3.5-9b`).

## 1. Start sharing the model

In the kit menu choose option 7, "Share API with n8n". Keep that window open.
It prints an API key and a Base URL such as `http://<laptop-ip>:8080/v1`.
The model you shared must match the model id used in the workflow (see step 4).

## 2. Create the credential in n8n

1. Credentials > Create credential > **OpenAI**.
2. API Key: the key printed by the kit.
3. Base URL: the URL printed by the kit (ends in `/v1`).
4. Name it `llm-kit (laptop)` and save.

## 3. Import the workflows

1. Workflows > top-right menu > **Import from File**.
2. Pick `llm-kit-chat-agent.json`, then repeat for `llm-kit-summarize-webhook.json`.
3. Open each workflow, click the **OpenAI Chat Model** node and choose the
   `llm-kit (laptop)` credential if it is not already selected.

## 4. Change the model if you want

In the **OpenAI Chat Model** node, the Model field is set "By ID". Replace the id
with the one you are sharing, for example `granite-4.2-8b`, `qwen3.5-9b` or
`qwen3.5-4b`.

## 5. Activate and use

- Chat agent: click **Open chat** in the editor (or activate it and use the chat link from the trigger node).
- Summarize webhook: switch the workflow to **Active** (Published), then send a POST.

curl (Linux/macOS, or Windows with curl.exe):

    curl -X POST http://localhost:5678/webhook/llm-kit-summarize \
      -H "Content-Type: application/json" \
      -d '{"text": "Paste the text you want summarized here."}'

PowerShell:

    Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/llm-kit-summarize `
      -ContentType "application/json" `
      -Body '{"text": "Paste the text you want summarized here."}'

The answer looks like:

    {"summary": "- first point\n- second point\n- third point"}

Replace `localhost:5678` with your n8n address. While testing without activating,
use the "Test URL" shown in the Webhook node (`/webhook-test/...`) after clicking
"Listen for test event".

Small models on a laptop can take a while to answer; long texts take longer.
