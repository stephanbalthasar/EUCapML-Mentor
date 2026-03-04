# Privacy & AI Transparency Notice — EUCapML Mentor
Effective date: 2026-03-04

This notice explains how the EUCapML Mentor ("App") functions, how it uses Artificial Intelligence ("AI") and Large Language Models ("LLMs"), and which external providers are involved ("Third Parties") that are outside the control of the App. 

1) Contact: The code for the App was created by Stephan Balthasar. The contact address for data / AI questions and exercising rights is: stephan.balthasar@uni-bayreuth.de.

2) Quick summary (what happens when you use the App): The App allows users to view sample exam questions from past years, submit answers of their own, and receive instant feedback on their submissions that the App creates through the use of AI and LLMs. The App is hosted by Streamlit. When users enter text in the app (e.g., answers on exam questions, chat messages), the App forwards that text to an external LLM provider (Groq) and runs web retrieval via Streamlit-hosted requests. On that basis, the App creates feedback that it then displays on screen.

3) AI / LLM usage: Provider & endpoint used in code: Groq — calls to https://api.groq.com/openai/v1/chat/completions. UI model options: "llama-3.1-8b-instant", "llama-3.3-70b-versatile". Purpose: generate feedback, extract structured issues, enforce consistency with an authoritative model answer, and power an interactive tutor chat. Inputs sent: user-entered free text (prompts, student answers, chat messages) plus internal context (selected model answer slice and retrieved source snippets). Outputs: generated textual feedback (<=400 words), structured JSON (extracted issues), and chat replies. Human oversight: outputs are shown to human users (no automatic external actions).

4) Data collection, use and retention by the App:
- Data collection: By default, the App does not collect any personal data whatsoever. The App does not need personal data of any type to function and does not prompt users to submit personal data of any kind. Accordingly, users must not paste personal data or sensitive categories into any text field as the app passes on user input to Third Parties (Streamlit, Groq, etc.) outside the control of the App. Third Parties such as the provider hosting the app (Streamlit) and the providers used for feedback generation (Groq) are independent controllers and may log, retain, or otherwise process the transmitted data under their own policies.
- Data use: The App passes on user input to Third Parties to generate feedback.
- Data retention: The App does not persist user text in the repository; session state is ephemeral. The App records user events on a fully anonymous basis through a minimal event log (timestamp, event type) to a GitHub Gist for metrics/diagnostics. These logs cannot be linked to any individual user and do NOT contain student answer text.
- GDPR: As the App does not collect personal data, GDPR requirements do not apply. The basis for processing user-submitted non-sensitive input is the performance of the requested service and legitimate interest (educational feedback). 

5) Third Parties, transfers & retention:
The App uses Third Parties outside its control, inter alia, for 
- Hosting: Streamlit/Snowflake (streamlit.io) — may collect session/connection metadata (IP, telemetry) under their policy. See https://streamlit.io/privacy. 
- LLM: Groq — Zero Data Retention is requested, but where this fails to be implemented for any reason, Groq may log/retain inputs/outputs under their policy. See  https://groq.com/privacy-policy. 
- Logs: GitHub Gist stores CSV logs (timestamp, event, role). Gist visibility and retention are governed by GitHub. See https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement
- International transfers: Groq and Streamlit may process data outside the EEA.

6) Data subject rights:
By design, the App does not collect or process personal data, and accordingly, the App is outside the scope of GDPR rules. Enquiries in relation to GDPR rights can be sent to the email address given in section 1 above, and complaints can be logged with competent supervisory authorities. Third parties may process personal data (e.g., IP addresses) under their own privacy policies (see section 5 above).

7) Liability & accuracy:
The App uses LLMs and web retrieval. Generated answers may be incorrect or incomplete. No liability is accepted under any circumstances. App feedback must not be read as an indicator for grades in a real examination. Users must exercise professional judgment as regards App output.

8) Changes:
This notice is subject to updates in the discretion of the owner of the App (see section 1 above).
