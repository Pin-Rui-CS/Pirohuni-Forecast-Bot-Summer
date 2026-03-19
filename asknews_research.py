"""
asknews_research.py (FIXED VERSION)

Standalone module for AskNews research with rate limiting.
This version fixes the infinite waiting issue by using proper thread locks.

Import this module in your forecasting_bot.py:
    from asknews_research import call_asknews_rate_limited, call_asknews_fast

Rate limits (Metaculus free tier):
- 1 request per 20 seconds
- 1 concurrent request at a time
"""

import os
import time
import threading
from datetime import datetime
from typing import Optional
from asknews_sdk import AskNewsSDK
from asknews_sdk.errors import RateLimitExceededError
import dotenv

dotenv.load_dotenv()

# Get API credentials from environment
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
ASKNEWS_MIN_SECONDS_BETWEEN_CALLS = 20.0

# Global variables for rate limiting
_last_asknews_call_time = None
_asknews_lock = threading.Lock()  # Use proper threading lock instead of boolean


def wait_for_rate_limit(min_seconds_between_calls: float = ASKNEWS_MIN_SECONDS_BETWEEN_CALLS):
    """
    Ensures we wait at least min_seconds_between_calls since the last API call.
    
    Args:
        min_seconds_between_calls: Minimum seconds to wait (20 for Metaculus free tier)
    """
    global _last_asknews_call_time
    
    if _last_asknews_call_time is not None:
        elapsed = time.time() - _last_asknews_call_time
        if elapsed < min_seconds_between_calls:
            wait_time = min_seconds_between_calls - elapsed
            print(f"⏳ Rate limit: waiting {wait_time:.1f} seconds before next AskNews call...")
            time.sleep(wait_time)
    
    _last_asknews_call_time = time.time()


def call_asknews_rate_limited(question: str) -> str:
    """
    Call AskNews with proper rate limiting for Metaculus free tier.
    
    Makes 2 API calls with proper spacing:
    1. Hot news (latest 48 hours)
    2. Historical news (past 60 days)
    
    Args:
        question: The forecasting question to research
    
    Returns:
        Formatted string containing relevant news articles
    
    Total time: ~40 seconds (20s between calls + API response time)
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        return "AskNews credentials not found in environment variables."
    
    # Acquire lock - this ensures only one request at a time
    # Using 'with' ensures lock is ALWAYS released, even if there's an error
    with _asknews_lock:
        try:
            print(f"🔍 Starting AskNews research for: {question[:60]}...")
            
            ask = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID,
                client_secret=ASKNEWS_SECRET,
                scopes=set(["news"])
            )
            
            formatted_articles = "Here are the relevant news articles:\n\n"
            
            # FIRST API CALL: Hot news
            hot_articles = None
            try:
                wait_for_rate_limit(min_seconds_between_calls=ASKNEWS_MIN_SECONDS_BETWEEN_CALLS)
                print("📡 API Call 1/2: Fetching latest news (past 48 hours)...")
                
                hot_response = ask.news.search_news(
                    query=question,
                    n_articles=6,
                    return_type="both",
                    strategy="latest news",
                )
                
                hot_articles = hot_response.as_dicts
                print(f"✓ Got {len(hot_articles) if hot_articles else 0} hot articles")
                
                if hot_articles:
                    hot_articles = [article.model_dump() for article in hot_articles]
                    hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

                    for article in hot_articles:
                        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                        key_points = article.get("key_points") or []
                        content = "\n".join(f"- {pt}" for pt in key_points) if key_points else article.get("summary", "")
                        formatted_articles += (
                            f"**{article['eng_title']}**\n"
                            f"{content}\n"
                            f"Original language: {article['language']}\n"
                            f"Publish date: {pub_date}\n"
                            f"Source:[{article['source_id']}]({article['article_url']})\n\n"
                        )
            
            except RateLimitExceededError as e:
                print(f"⚠️  Rate limit hit on hot news: {str(e)}")
                formatted_articles += "⚠️  Rate limit reached on latest news search.\n\n"
            
            # SECOND API CALL: Historical news
            historical_articles = None
            try:
                wait_for_rate_limit(min_seconds_between_calls=ASKNEWS_MIN_SECONDS_BETWEEN_CALLS)
                print("📡 API Call 2/2: Fetching historical news (past 60 days)...")
                
                historical_response = ask.news.search_news(
                    query=question,
                    n_articles=10,
                    return_type="both",
                    strategy="news knowledge",
                )
                
                historical_articles = historical_response.as_dicts
                print(f"✓ Got {len(historical_articles) if historical_articles else 0} historical articles")
                
                if historical_articles:
                    historical_articles = [article.model_dump() for article in historical_articles]
                    historical_articles = sorted(
                        historical_articles,
                        key=lambda x: x["pub_date"],
                        reverse=True
                    )

                    for article in historical_articles:
                        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                        key_points = article.get("key_points") or []
                        content = "\n".join(f"- {pt}" for pt in key_points) if key_points else article.get("summary", "")
                        formatted_articles += (
                            f"**{article['eng_title']}**\n"
                            f"{content}\n"
                            f"Original language: {article['language']}\n"
                            f"Publish date: {pub_date}\n"
                            f"Source:[{article['source_id']}]({article['article_url']})\n\n"
                        )
            
            except RateLimitExceededError as e:
                print(f"⚠️  Rate limit hit on historical news: {str(e)}")
                formatted_articles += "⚠️  Rate limit reached on historical news search.\n\n"
            
            # Check if we got any results
            if not hot_articles and not historical_articles:
                return "AskNews rate limit exceeded or no articles found. Please wait 20 seconds and try again."
            
            print("✓ AskNews research completed successfully!")
            return formatted_articles
        
        except ImportError:
            return "AskNews SDK not installed. Run: pip install asknews-sdk"
        
        except Exception as e:
            error_msg = f"AskNews error: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg
        
        # Note: No 'finally' needed - the 'with' statement automatically releases the lock


def call_asknews_fast(question: str, max_wait: float = 30.0) -> str:
    """
    Faster version that only gets hot news (1 API call instead of 2).
    
    Use this when you want faster results and don't need historical context.
    
    Args:
        question: The forecasting question
        max_wait: Maximum seconds to wait for rate limit (default 30)
    
    Returns:
        Formatted string containing relevant news articles
    
    Total time: ~20 seconds + API response time
    """
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        return "AskNews credentials not found in environment variables."
    
    # Acquire lock with automatic release
    with _asknews_lock:
        try:
            print(f"🔍 Starting fast AskNews research (hot news only)...")
            
            ask = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID,
                client_secret=ASKNEWS_SECRET,
                scopes=set(["news"])
            )
            
            # Check if we need to wait
            global _last_asknews_call_time
            if _last_asknews_call_time is not None:
                elapsed = time.time() - _last_asknews_call_time
                if elapsed < ASKNEWS_MIN_SECONDS_BETWEEN_CALLS:
                    wait_time = ASKNEWS_MIN_SECONDS_BETWEEN_CALLS - elapsed
                    if wait_time > max_wait:
                        return f"Rate limit: need to wait {wait_time:.0f}s (exceeds max_wait={max_wait}s)"
            
            wait_for_rate_limit(min_seconds_between_calls=ASKNEWS_MIN_SECONDS_BETWEEN_CALLS)
            print("📡 Fetching latest news...")
            
            hot_response = ask.news.search_news(
                query=question,
                n_articles=6,
                return_type="both",
                strategy="latest news",
            )
            
            hot_articles = hot_response.as_dicts
            print(f"✓ Got {len(hot_articles) if hot_articles else 0} articles")
            
            if not hot_articles:
                return "No recent articles found.\n"
            
            formatted_articles = "Here are the relevant news articles:\n\n"
            hot_articles = [article.model_dump() for article in hot_articles]
            hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                key_points = article.get("key_points") or []
                content = "\n".join(f"- {pt}" for pt in key_points) if key_points else article.get("summary", "")
                formatted_articles += (
                    f"**{article['eng_title']}**\n"
                    f"{content}\n"
                    f"Original language: {article['language']}\n"
                    f"Publish date: {pub_date}\n"
                    f"Source:[{article['source_id']}]({article['article_url']})\n\n"
                )
            
            print("✓ Fast AskNews research completed!")
            return formatted_articles
        
        except ImportError:
            return "AskNews SDK not installed. Run: pip install asknews-sdk"
        
        except RateLimitExceededError as e:
            print(f"⚠️  Rate limit exceeded: {str(e)}")
            return "Rate limit exceeded. Please wait 20 seconds."
        
        except Exception as e:
            error_msg = f"AskNews error: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg


def batch_asknews_research(questions: list[str], use_fast_mode: bool = False) -> dict[str, str]:
    """
    Run research on multiple questions with proper rate limiting.
    
    Args:
        questions: List of questions to research
        use_fast_mode: If True, only get hot news (faster, 1 call per question)
                      If False, get both hot and historical (slower, 2 calls per question)
    
    Returns:
        Dictionary mapping questions to their research results
    
    Time estimate:
        Fast mode: ~20 seconds per question
        Full mode: ~40 seconds per question
    """
    results = {}
    total_questions = len(questions)
    
    print(f"\n{'='*80}")
    print(f"BATCH RESEARCH: {total_questions} questions")
    print(f"Mode: {'Fast (hot news only)' if use_fast_mode else 'Full (hot + historical)'}")
    print(f"Estimated time: {(20 if use_fast_mode else 40) * total_questions} seconds")
    print(f"{'='*80}\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{total_questions}]")
        
        if use_fast_mode:
            research = call_asknews_fast(question)
        else:
            research = call_asknews_rate_limited(question)
        
        results[question] = research
        
        # Show progress
        remaining = total_questions - i
        if remaining > 0:
            est_time = (20 if use_fast_mode else 40) * remaining
            print(f"✓ Completed. {remaining} questions remaining (~{est_time}s)")
    
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE: Researched {total_questions} questions")
    print(f"{'='*80}\n")
    
    return results


# For testing this module directly
if __name__ == "__main__":
    print("Testing AskNews Research Module (FIXED VERSION)")
    print("=" * 80)
    
    # Check if credentials are available
    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        print("❌ AskNews credentials not found in environment variables")
        print("Please set ASKNEWS_CLIENT_ID and ASKNEWS_SECRET in your .env file")
    else:
        print("✓ AskNews credentials found")
        
        # Test with a sample question
        test_question = "What is the current status of AI development?"
        print(f"\nTesting with question: {test_question}")
        print("-" * 80)
        
        result = call_asknews_fast(test_question)
        print("\nResult:")
        print(result[:500] if len(result) > 500 else result)
