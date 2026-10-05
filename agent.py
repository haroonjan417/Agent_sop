from crewai import Agent, Task, Crew, Process, LLM
from tools import sop_search_rag

# Workaround for Groq rejecting CrewAI's cache breakpoint markers (version dependent).
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except Exception:
    pass


def run_sop_agent(user_incident: str, api_key: str, model_name: str = "groq/openai/gpt-oss-120b"):
    """Runs the single SOP agent: it retrieves SOPs via RAG, then writes the full action package."""
    llm = LLM(model=model_name, api_key=api_key)

    sop_agent = Agent(
        role="Operations SOP & Execution Manager",
        goal=(
            "Analyze operational incidents against the company's SOPs and produce an "
            "actionable, cited resolution package for manager approval."
        ),
        backstory=(
            "An expert operations and compliance lead in logistics and warehousing. "
            "You rely strictly on retrieved company SOPs, never on guesses, and you always "
            "cite the SOP document each rule comes from."
        ),
        tools=[sop_search_rag],
        llm=llm,
        allow_delegation=False,
        max_iter=8,
        verbose=True,
    )

    sop_task = Task(
        description=(
            f"An operational incident was reported:\n'{user_incident}'\n\n"
            "Follow this sequence:\n"
            "1. Use the 'SOP Policy Search Tool' to retrieve the SOP procedures relevant to this incident "
            "(search again with a different query if the first result is not relevant).\n"
            "2. Identify the exact SOP document name and section for every rule you rely on "
            "(use the '[Document Source: ...]' label from the results). If no relevant SOP is found, say so "
            "clearly instead of inventing procedures.\n"
            "3. Produce the final package below."
        ),
        expected_output=(
            "A clean markdown response containing:\n"
            "1) Executive Incident Summary (2-3 sentences)\n"
            "2) Cited SOP Document Name(s) & applicable rules, plus risks if steps are delayed\n"
            "3) Prioritized Action Checklist (High / Medium / Low)\n"
            "4) Pre-filled Communication Draft (e.g. supplier claim email, maintenance ticket, or incident log)\n"
            "5) Status flag: 'PENDING MANAGER APPROVAL'"
        ),
        agent=sop_agent,
    )

    crew = Crew(agents=[sop_agent], tasks=[sop_task], process=Process.sequential)
    return crew.kickoff()
