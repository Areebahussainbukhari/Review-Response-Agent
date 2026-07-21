# ---------------- Imports ----------------
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import os
import streamit  as st

# ---------------- Load environment variables ----------------
load_dotenv()

# ---------------- LLM Setup ----------------
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# ---------------- Review Data Model ----------------
class Review(BaseModel):
    id: str = Field(description="Unique review ID")
    platform: str = Field(description="Platform name (Google, Facebook, Instagram, Website)")
    business_name: str = Field(description="Business name")
    reviewer_name: str = Field(description="Name of reviewer")
    review_text: str = Field(description="The review content")
    rating: int = Field(description="Star rating 1-5")
    date: str = Field(description="Date of review")
    already_replied: bool = Field(default=False, description="Whether business already replied")

# ---------------- Sample Review Data ----------------
sample_reviews = [
    Review(
        id="review_001",
        platform="Google",
        business_name="Sample Cafe",
        reviewer_name="Ali Khan",
        review_text="Amazing service, the staff was so friendly and my order arrived early!",
        rating=5,
        date="2026-07-12",
        already_replied=False
    ),
    Review(
        id="review_002",
        platform="Facebook",
        business_name="Sample Cafe",
        reviewer_name="Sara Ahmed",
        review_text="Very disappointed, waited over an hour and the food was cold when it arrived.",
        rating=2,
        date="2026-07-11",
        already_replied=False
    ),
    Review(
        id="review_003",
        platform="Instagram",
        business_name="Sample Cafe",
        reviewer_name="Hassan Malik",
        review_text="It was okay, nothing special but nothing bad either.",
        rating=3,
        date="2026-07-10",
        already_replied=False
    )
]

# ---------------- Agents ----------------
sentiment_analyzer = Agent(
    role="Sentiment Analyzer",
    goal="Analyze review sentiment and identify key issues or highlights",
    backstory="Expert in understanding customer sentiment, identifies the core concern in reviews",
    llm=llm,
    verbose=False
)

response_writer = Agent(
    role="Response Writer",
    goal="Write personalized replies matching the tone of each review",
    backstory="Expert in customer communication, writes warm replies for positive reviews and empathetic solution focused replies for negative ones",
    llm=llm,
    verbose=False
)

reviewer = Agent(
    role="Reply Reviewer",
    goal="Polish replies so they sound natural, professional and ready to post",
    backstory="Expert editor who ensures replies sound human, on brand and appropriate for public posting",
    llm=llm,
    verbose=False
)

# ---------------- Core Processing Function ----------------
def process_review(review: Review):
    analyze_task = Task(
        description=f"Analyze this {review.platform} review:\n{review.review_text}\nIdentify sentiment and key issue.",
        expected_output="Sentiment and key issue summary",
        agent=sentiment_analyzer
    )

    write_task = Task(
        description=f"Write a reply on behalf of {review.business_name} to this {review.platform} review:\n{review.review_text}",
        expected_output="Draft reply to the review",
        agent=response_writer
    )

    polish_task = Task(
        description=f"Polish this reply for {review.platform} so it sounds natural and professional",
        expected_output="Final polished reply ready to post",
        agent=reviewer
    )

    crew = Crew(
        agents=[sentiment_analyzer, response_writer, reviewer],
        tasks=[analyze_task, write_task, polish_task]
    )

    try:
        result = crew.kickoff()
        return {
            "review_id": review.id,
            "platform": review.platform,
            "business": review.business_name,
            "reviewer": review.reviewer_name,
            "review_text": review.review_text,
            "rating": review.rating,
            "date": review.date,
            "generated_reply": str(result),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "Pending Approval"
        }
    except Exception as e:
        return {
            "review_id": review.id,
            "platform": review.platform,
            "error": str(e)
        }

# ---------------- Batch Processing Function ----------------
def process_all_unreplied_reviews(reviews_list):
    results = []
    for review in reviews_list:
        if not review.already_replied:
            print(f"Processing review {review.id} from {review.platform}...")
            output = process_review(review)
            results.append(output)
    return pd.DataFrame(results)

# ---------------- Run ----------------
if __name__ == "__main__":
    df_results = process_all_unreplied_reviews(sample_reviews)

    print("\n=== REVIEW RESPONSE RESULTS ===\n")
    print(df_results)

    # Save results locally
    df_results.to_csv("review_responses_pending_approval.csv", index=False)
    df_results.to_json("review_responses.json", orient="records", indent=2)

    print("\nSaved review responses to:")
    print("  - review_responses_pending_approval.csv")
    print("  - review_responses.json")
