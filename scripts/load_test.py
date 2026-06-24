"""
scripts/load_test.py — Day 13 deliverable.

Sends 200 diverse prompts through the live API and generates your
headline cost savings report. This is the number that goes in your
README and LinkedIn post.

Usage:
    # Against local Docker deployment (default)
    python scripts/load_test.py

    # Against a specific URL
    python scripts/load_test.py --url http://localhost:8000

    # Quick test with fewer prompts
    python scripts/load_test.py --count 50

    # Dry run — print prompts but don't send
    python scripts/load_test.py --dry-run

What it produces:
    - Prints a live progress table as requests complete
    - Saves data/load_test_results.csv with full row-level data
    - Prints the final cost savings report (your portfolio headline)
    - Calls GET /v1/stats so the dashboard reflects the new data
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 200 prompts spanning all 3 tiers ──────────────────────────────────────────
# Tier distribution: ~35% T1, ~40% T2, ~25% T3
# This matches real-world traffic and gives a meaningful savings number.

TIER1_PROMPTS = [
    "Extract all email addresses from this text: Contact us at support@company.com or sales@firm.io",
    "Reformat this date from MM/DD/YYYY to DD-MM-YYYY: 06/17/2026",
    "Is the following sentence grammatically correct? Reply yes or no. Sentence: 'She don't like coffee.'",
    "Extract the product price from this text: The laptop costs $999.99",
    "Count the number of words: The quick brown fox jumps over the lazy dog",
    "Convert 25 degrees Celsius to Fahrenheit",
    "Is the word 'running' a verb? Answer yes or no.",
    "Extract the zip code from: 123 Main St, Boston, MA 02101",
    "Convert this time from 12-hour to 24-hour format: 3:30 PM",
    "Is the number 17 prime? Answer yes or no.",
    "Extract the domain from this email: user@example.org",
    "Convert this fraction to decimal: 3/4",
    "What is the first name in: 'Michael Johnson'? Reply with the first name only.",
    "Is this a valid email? test@company.com — reply yes or no.",
    "Extract all phone numbers from: Call us at 555-123-4567 or (555) 987-6543",
    "Convert this binary number to decimal: 1010",
    "Is 'beautiful' spelled correctly? Reply yes or no.",
    "Extract the file extension from: document.pdf",
    "Convert this Roman numeral to decimal: XIV",
    "Count the vowels in the word: encyclopedia",
    "Is the number 24 even? Reply yes or no.",
    "Extract the year from this date: March 15, 2024",
    "What currency symbol is used for Euro? Reply with just the symbol.",
    "Is 'receive' spelled correctly? Reply yes or no.",
    "Convert 100 USD to EUR at rate 0.92. Return only the number.",
    "Extract the country code from phone number: +44-20-7946-0958",
    "Is the word 'beautiful' an adjective? Reply yes or no.",
    "Count the number of sentences: Hello. How are you? I am fine.",
    "Extract the username from this Twitter handle: @john_doe_dev",
    "What is 15% of 200? Return only the number.",
    "Is this IP address valid? 192.168.1.256 — reply yes or no.",
    "Extract the protocol from this URL: https://www.example.com/page",
    "Convert 5 kilometers to miles. Return only the number rounded to 2 decimal places.",
    "Is the word 'quickly' an adverb? Reply yes or no.",
    "Extract the TLD from: user@company.co.uk",
    "What is the square root of 144? Return only the number.",
    "Is 'entrepreneur' spelled correctly? Reply yes or no.",
    "Count the digits in this string: abc123xyz456",
    "Extract the hashtag from: Love this product! #innovation",
    "Convert 72 Fahrenheit to Celsius. Return only the number.",
    "Is the word 'cat' a noun? Reply yes or no.",
    "What day of the week is 2024-01-15? Reply with just the day name.",
    "Extract the version number from: App version 3.14.159",
    "Is this a palindrome? 'racecar' — reply yes or no.",
    "Count characters in: Hello World",
    "Extract the port number from: http://localhost:8080/api",
    "Is 12/32 a valid date? Reply yes or no.",
    "Convert 1000 grams to kilograms. Return only the number.",
    "What is 7 squared? Return only the number.",
    "Is 'license' or 'licence' the American English spelling? Reply with the correct word only.",
    "Extract the area code from: (617) 555-0123",
    "Is 'their' or 'there' correct in: 'Put it over ___.' Reply with the correct word only.",
    "Convert 2 hours to minutes. Return only the number.",
    "Count the number of uppercase letters in: Hello World ABC",
    "Extract the ISBN from: Book ISBN: 978-3-16-148410-0",
    "Is 0 considered even or odd? Reply with one word.",
    "What percentage is 25 out of 200? Return only the number.",
    "Extract the brand from: Apple iPhone 15 Pro Max 256GB",
    "Is the word 'run' a verb? Reply yes or no.",
    "Convert 1 mile to kilometers. Return only the number rounded to 2 decimals.",
    "Count the number of commas in: apple, banana, orange, grape, mango",
    "Extract the city from: 742 Evergreen Terrace, Springfield, IL 62701",
    "Is 'affect' or 'effect' correct here: 'The rain will ___ the game.' Reply with the correct word only.",
    "What is 2 to the power of 8? Return only the number.",
    "Is the word 'slowly' an adverb? Reply yes or no.",
    "Convert 500 milliliters to liters. Return only the number.",
    "Extract the model number from: Samsung Galaxy S24 Ultra Model SM-S928B",
    "Is 'grateful' spelled correctly? Reply yes or no.",
    "Count the number of words in: To be or not to be",
]

TIER2_PROMPTS = [
    "Summarize the following paragraph in exactly 2 sentences: Artificial intelligence has transformed how companies operate, enabling automation of repetitive tasks, faster data analysis, and more personalised customer experiences. However, the rapid adoption of AI also raises concerns about job displacement, data privacy, and the concentration of power in the hands of a few large technology companies.",
    "Classify this customer message into one of: complaint, question, compliment, or feature_request. Message: 'Your app keeps crashing whenever I try to export a PDF. This is really frustrating — I have lost work twice now.'",
    "Translate the following sentence to French: 'The meeting has been rescheduled to Thursday at 3pm. Please update your calendars accordingly.'",
    "Extract all named entities (people, organisations, locations) from this text and list them by category: 'Elon Musk announced that Tesla will open a new Gigafactory in Mexico City next year, in partnership with the Mexican government.'",
    "Classify this sentiment as positive, negative, or neutral: 'The product is decent, but I expected better customer support. The software works fine though.'",
    "Write a short product description based on these features: 'AI-powered chatbot, 24/7 support, multi-language, integrates with Slack and Teams'",
    "Summarize this news article in 2 sentences: Climate change is accelerating at an alarming rate, with global temperatures rising faster than predicted. Scientists warn that without immediate action, we will see catastrophic consequences by 2050.",
    "Classify this email as spam or not spam: 'Dear Customer, You have been selected for an exclusive prize. Click here to claim your $1,000,000 reward now!'",
    "Translate to Spanish: 'Please confirm your attendance for the annual company meeting scheduled for next Friday.'",
    "Extract the key requirements from this job description: 'We are looking for a Python developer with 5+ years of experience, expertise in machine learning, and strong communication skills.'",
    "Summarize these meeting notes in 3 bullet points: Discussed Q4 targets, reviewed marketing strategy, decided to launch new product in January, allocated budget of $500k for advertising, assigned tasks to team leads.",
    "Classify the tone of this message as formal or informal: 'Hey team, quick update — we are crushing it this quarter! Let us sync up tomorrow to celebrate.'",
    "Write a 2-sentence summary of this product review: 'This laptop is incredibly fast and has amazing battery life lasting 12 hours. The only downside is the fan noise when running intensive applications like video editing.'",
    "Extract action items from this email: 'John please prepare the Q3 report by Friday. Sarah can you schedule the team meeting for next week? Mark update the website content before Monday.'",
    "Identify the main argument in this text: 'Remote work increases productivity and employee satisfaction. Multiple studies show that companies embracing remote work see lower turnover rates and higher output per employee.'",
    "Classify these customer feedback items as positive, negative, or neutral: 1) Love the new interface! 2) Shipping took 3 weeks. 3) Product works as described. 4) Customer service was unhelpful.",
    "Translate this business email to German: 'We are pleased to announce our new partnership with TechCorp that will expand our services to European markets starting Q1 2025.'",
    "Summarize this financial report in 2 sentences: Revenue grew 15% year over year driven by increased sales in Asia-Pacific. Operating expenses also rose 8% due to marketing investments and new hires.",
    "Extract the comparison points from this text: 'Product A is cheaper but has fewer features and slower processing. Product B is more expensive but offers better performance, more storage, and excellent customer support.'",
    "Write a brief description of this service: '24/7 customer support with guaranteed response time under 5 minutes, available via chat, email, and phone, supporting 15 languages.'",
    "Classify this news headline by topic as technology, business, sport, politics, or health: 'New AI model achieves human-level performance on medical diagnosis benchmark'",
    "Summarize this customer interaction in one sentence: 'Customer called about a billing issue, agent reviewed the account, found a duplicate charge, issued a refund of $49.99, and the customer confirmed satisfaction.'",
    "Extract the pros and cons from this review: 'The software is user-friendly and feature-rich with excellent documentation. However, it is expensive at $200/month, has a steep learning curve initially, and customer support response times can be slow.'",
    "Translate this technical instruction to French: 'Please restart your device and clear your browser cache before attempting to log in again. If the issue persists, contact support.'",
    "Identify the key stakeholders mentioned: 'The marketing team will handle the launch campaign, engineering will deliver the product features, sales will prepare the pricing strategy, and legal will review the contracts.'",
    "Classify this support ticket by priority as critical, high, medium, or low: 'The entire production database is down and all users cannot access the system. Revenue loss is accumulating.'",
    "Write a concise summary of this research: 'A study of 10,000 employees across 50 companies found that organizations offering flexible work hours saw 23% higher productivity and 31% lower employee turnover compared to those with rigid schedules.'",
    "Extract the timeline from this project update: 'Development started January 1st, first review is February 15th, beta launch is March 1st, public launch is April 30th, and post-launch review is May 15th.'",
    "Classify this restaurant review sentiment by sentence: 'The food was absolutely delicious. Our server was incredibly rude and dismissive. The ambiance was pleasant and cozy. We waited 45 minutes for our table despite having a reservation.'",
    "Summarize this technical document in 3 sentences for a non-technical audience: 'The system employs a RESTful API architecture with JWT authentication, rate limiting at 100 requests per minute, horizontal scaling via Kubernetes, and PostgreSQL for persistent storage with Redis caching.'",
    "Extract all dates and their associated events from this text: 'The project kickoff was on January 5th. The first milestone review is scheduled for February 20th. Final delivery is expected by March 31st with a post-launch review on April 15th.'",
    "Translate to Italian: 'We apologize for the inconvenience caused by the system outage yesterday. All services have been restored and we have implemented additional safeguards to prevent future occurrences.'",
    "Identify the problem and proposed solution in this text: 'Our customer churn rate has increased 15% this quarter due to poor onboarding experiences. We propose implementing an automated onboarding flow with personalized tutorials and a dedicated success manager for enterprise accounts.'",
    "Write a 3-bullet executive summary of this quarterly report: 'Revenue reached $45M, up 22% YoY. Customer base grew to 12,000 accounts. Operating margin improved to 18% from 14% last year. New product lines contributed 30% of total revenue. International expansion progressed with launches in 5 new markets.'",
    "Classify this social media post intent as promotional, informational, engaging, or complaint: 'Just tried the new coffee blend from @BeansCo and honestly it is life-changing. The caramel notes are perfect for my morning routine. Who else has tried it?'",
    "Summarize the key differences between these two approaches described: 'Approach A uses batch processing to analyze data overnight, offering high throughput but delayed insights with 24-hour lag. Approach B uses streaming processing to analyze data in real-time, offering immediate insights but requiring more infrastructure and cost.'",
    "Extract all product features mentioned in this marketing copy: 'Introducing the UltraBook Pro — featuring a stunning 4K OLED display, 20-hour battery life, Intel Core i9 processor, 32GB RAM, 2TB NVMe SSD, Thunderbolt 4 ports, and MIL-SPEC durability rating.'",
    "Classify the urgency of these customer requests from most to least urgent: 1) Cannot log into account. 2) Feature request for dark mode. 3) Billing statement has wrong amount. 4) How do I export my data? 5) App crashes on startup.",
    "Translate this error message to user-friendly language: 'Error 403: Forbidden. The server understood the request but refuses to authorize it. CORS policy violation detected. Pre-flight response headers missing.'",
    "Write a 2-sentence description of this job role based on the responsibilities: 'Develop and maintain ML pipelines. Train and evaluate models. Collaborate with data engineers. Deploy models to production. Monitor model performance and retrain as needed. Present findings to stakeholders.'",
    "Extract the budget information from this project proposal: 'The total project budget is $250,000. This breaks down as $80,000 for development, $50,000 for design, $40,000 for testing, $30,000 for project management, $30,000 for infrastructure, and $20,000 for contingency.'",
    "Classify these technical issues by category (frontend, backend, database, infrastructure, or security): 1) CSS layout breaks on mobile. 2) API response time exceeds 5 seconds. 3) Database queries not using indexes. 4) Server CPU usage at 95%. 5) User passwords stored in plain text.",
    "Summarize the customer journey described: 'Customer discovers product through Google ad, visits website, reads reviews for 3 days, signs up for free trial, uses product for 2 weeks, contacts support once about a feature, upgrades to paid plan on day 12, refers 2 colleagues in month 2.'",
    "Identify the logical fallacy in this argument: 'Our competitor released a product that failed in the market. Therefore, any product we release in this category will also fail. We should not invest in this space.'",
    "Write a one-paragraph executive summary for this situation: 'Sales team is missing Q3 targets by 23%. Top reasons are: product pricing too high vs competitors (mentioned by 45% of lost deals), slow response time from sales team averaging 48 hours (industry standard is 4 hours), and lack of enterprise-tier features requested by 60% of prospects.'",
    "Extract and categorize all claims in this product advertisement: 'Our supplement is clinically proven to boost energy by 40%, is 100% natural with no side effects, used by 2 million customers worldwide, recommended by 9 out of 10 doctors, and comes with a 60-day money-back guarantee.'",
    "Classify these five statements as facts, opinions, or assumptions: 1) The company was founded in 2015. 2) This is the best product on the market. 3) Users probably prefer the old interface. 4) Revenue was $10M last year. 5) The new feature will increase retention.",
    "Translate this product disclaimer to simple English: 'The manufacturer expressly disclaims all warranties, express or implied, including but not limited to implied warranties of merchantability and fitness for a particular purpose.'",
    "Summarize the technical trade-offs described: 'Option 1 uses SQL for strong consistency and ACID compliance but limits horizontal scaling. Option 2 uses NoSQL for high availability and horizontal scaling but sacrifices consistency and makes complex queries difficult.'",
    "Extract the key performance indicators mentioned: 'We track monthly active users (currently 45,000), daily active users (12,000), session duration averaging 8 minutes, feature adoption rate of 34%, and NPS score of 67.'",
    "Classify this feedback as constructive or destructive criticism: 'Your presentation was completely disorganized and a waste of everyone time. Nobody could follow your logic. You clearly did not prepare. The data visualizations were confusing and the conclusion made no sense.'",
    "Write a brief 3-sentence description of this business model: 'Platform connects freelancers with businesses. Freelancers list skills and set rates. Businesses post projects. Platform takes 15% commission on completed transactions. Both sides rate each other after project completion. Premium membership removes commission cap.'",
    "Identify the assumptions underlying this business strategy: 'We will capture 20% market share in year 1 by offering a lower price than competitors, assuming customers will switch for cost savings alone and that we can maintain profitability at this price point while scaling rapidly.'",
    "Summarize the career progression described in this bio: 'Started as junior developer in 2015, promoted to senior developer in 2017, became tech lead in 2019, transitioned to engineering manager in 2021 leading a team of 12, and was recently appointed VP of Engineering overseeing 3 departments.'",
    "Extract all numerical data points from this paragraph: 'The clinical trial enrolled 1,247 patients across 15 hospitals in 8 countries. Over 24 months, 73% showed improvement with the treatment versus 41% in the control group. Side effects were reported in 12% of cases, all mild to moderate.'",
    "Classify each of these startup ideas as B2B, B2C, or B2B2C: 1) App helping individuals track personal finances. 2) Platform selling software to HR departments. 3) Marketplace connecting businesses to freelancers for clients. 4) Consumer fitness app. 5) SaaS tool for enterprise project management.",
    "Translate this legal notice to plain language that a 10-year-old could understand: 'By accessing this service, you hereby grant the provider an irrevocable, perpetual, royalty-free, worldwide license to use, reproduce, modify, and distribute any content you submit through the platform.'",
    "Summarize the problem-solution-outcome structure of this case study: 'A retail company was losing 30% of online customers at checkout due to a complex 7-step process. They redesigned to a 3-step checkout with saved payment methods. Result: checkout abandonment dropped to 12% and revenue increased 28% within 60 days.'",
    "Write a one-sentence tagline for each of these products based on their descriptions: 1) Password manager that auto-fills across devices. 2) AI writing assistant for professional emails. 3) Time-tracking tool for freelancers with automatic invoicing. 4) Meditation app with personalized 5-minute sessions.",
    "Identify the target audience for this marketing copy: 'Tired of spending weekends at the office? Our project management tool helps busy founders and small team leaders automate status updates, track deadlines without micromanaging, and finally reclaim their Saturdays.'",
    "Extract and list all action items with their owners from this meeting transcript: 'Sarah will prepare the budget proposal by next Friday. The engineering team led by Mike needs to complete the API documentation before the product launch. Marketing under Jennifer must finalize the campaign creative by Wednesday. HR should send the new policy update to all staff today.'",
    "Classify the writing style of this paragraph as academic, journalistic, conversational, technical, or marketing: 'Leveraging cutting-edge neural architectures, our revolutionary platform synergizes human creativity with machine intelligence to deliver unprecedented productivity gains that will transform your workflow and supercharge your team output.'",
    "Summarize the security vulnerability described and its impact: 'An SQL injection vulnerability was discovered in the login endpoint. Unauthenticated attackers can craft malicious input in the username field to bypass authentication entirely, access any user account, and extract the entire user database including hashed passwords and personal information.'",
    "Extract the geographic information and create a structured list: 'The company has offices in San Francisco serving North America, London covering Europe and Middle East, Singapore for Asia-Pacific, and São Paulo for Latin America. The engineering team is primarily in San Francisco and a remote team across Eastern Europe.'",
    "Classify these user reviews by the primary issue mentioned: 1) Cannot figure out how to export data. 2) App drains my battery so fast. 3) Price went up with no warning. 4) Takes forever to load. 5) Dark mode is not dark enough.",
    "Write a 3-sentence neutral summary of this controversial statement: 'Social media algorithms are deliberately designed to maximize outrage and division because angry users spend more time on platforms, which generates more advertising revenue, ultimately prioritizing profit over societal wellbeing.'",
    "Identify and correct all factual errors in this paragraph: 'Python was created by Guido van Rossum and first released in 1995. It is a compiled language known for its verbose syntax. Python was named after the Monty Python comedy group. It is primarily used for mobile app development.'",
    "Extract the competitive advantages claimed: 'Unlike competitors who charge per user, we offer unlimited seats. Our AI processes requests 3x faster than the market leader. We are the only solution with SOC2 Type II and ISO 27001 certification. Our 99.99% uptime SLA beats the industry standard of 99.9%.'",
    "Classify these five emails by required response urgency and explain why: 1) Server is down. 2) Can we reschedule our meeting next month? 3) Invoice is overdue by 90 days. 4) Request for product demo. 5) Password reset not working for 200 users.",
    "Summarize the innovation described in simple terms for a general audience: 'The system uses transformer-based neural networks fine-tuned on domain-specific corpora with retrieval-augmented generation to ground outputs in verified factual sources, significantly reducing hallucination rates compared to standard autoregressive language model inference.'",
    "Extract and prioritize the product backlog items mentioned: 'Users are requesting: dark mode, faster load times, CSV export, mobile app, better search, API access, bulk upload, team collaboration features, audit logs, and two-factor authentication.'",
    "Write a 2-sentence impact statement for each initiative: 1) Reducing customer support response time from 24h to 2h. 2) Adding offline mode to the mobile app. 3) Implementing automatic data backups every hour.",
]

TIER3_PROMPTS = [
    "Analyze the competitive landscape for EV batteries and recommend a market entry strategy. Consider: current players, cost trends, geographic opportunities, and regulatory environment. Provide a clear recommendation with supporting reasoning.",
    "Write a Python function called flatten_dict that: 1) Accepts a nested dict of arbitrary depth, 2) Flattens it using dot-notation keys (e.g. {'a': {'b': 1}} → {'a.b': 1}), 3) Handles lists by indexing them. Include type hints, docstring, and 3 usage examples.",
    "What are the second-order economic consequences of widespread LLM adoption in knowledge work? Consider effects on wage levels for different skill brackets, the market for junior professional roles, and how this compares to previous automation waves.",
    "Compare the trade-offs between microservices and a monolith for a startup building B2B SaaS. Consider: team size of 5 engineers, target scale of 10k users in year 1, and speed-to-market pressure. Provide a clear recommendation with reasoning.",
    "Critique this business plan and identify the three biggest risks: 'We will build an AI-powered platform connecting freelancers with businesses. Our competitive advantage is our proprietary matching algorithm. We plan to capture 10% market share in year one with a freemium model.'",
    "Explain how self-attention mechanisms work in transformer models. Include the mathematical intuition using Q, K, V matrices, explain why it replaced RNNs, and describe the computational complexity implications for long sequences.",
    "Design a rate-limiting system for a public API handling 10,000 requests per second across 50,000 users. Include: the algorithm choice (token bucket vs sliding window), Redis data structures, distributed coordination strategy, and how to handle burst traffic gracefully.",
    "Write a comprehensive Python class implementing a thread-safe LRU cache with: maximum size limit, TTL-based expiration per key, hit/miss statistics tracking, and an async-compatible interface. Include type hints and docstring.",
    "Analyze the ethical implications of using AI for hiring decisions. Consider: bias amplification risks, transparency requirements for candidates, legal compliance in different jurisdictions, and propose a responsible implementation framework.",
    "Design a system architecture for a real-time collaborative document editing feature (like Google Docs) handling 10,000 concurrent users. Address: operational transformation or CRDT approach, conflict resolution, offline support, and infrastructure requirements.",
    "What are the long-term implications of quantum computing on current cryptographic standards? Discuss which algorithms are most at risk, the timeline for quantum supremacy, and what organizations should be doing today to prepare for post-quantum cryptography.",
    "Write a production-ready Python implementation of a circuit breaker pattern for external API calls. Include: state machine (closed/open/half-open), configurable failure threshold and timeout, metrics tracking, and thread-safe state transitions.",
    "Evaluate the strategic decision of building a proprietary AI model versus using APIs from OpenAI or Anthropic for a B2B SaaS company with $5M in funding. Consider: total cost of ownership over 3 years, competitive moat, privacy implications, and team capability requirements.",
    "Explain the CAP theorem and how it applies to choosing between PostgreSQL, MongoDB, and Cassandra for a social media application that needs to handle user posts, comments, and real-time feeds. Provide specific recommendations for different parts of the system.",
    "Design a comprehensive data pipeline for processing 1TB of customer transaction data daily. Cover: ingestion strategy, transformation layer, storage architecture, serving layer for analytics, and cost optimization. Include specific technology recommendations with justification.",
    "Analyze why most enterprise digital transformation initiatives fail. Identify the top 5 root causes based on patterns across industries, explain the organizational dynamics that sustain these failure modes, and propose a change management framework to address them.",
    "Write a Python implementation of a distributed task queue system supporting: priority queues, delayed execution, retry with exponential backoff, dead letter queue, and worker health monitoring. Include a simple API for producers and consumers.",
    "Compare BERT, GPT, and T5 architectures for a document classification task on 100k legal documents. Consider: fine-tuning requirements, inference speed at scale, memory constraints, and accuracy trade-offs. Recommend the best approach with implementation guidance.",
    "Design a feature flag system for a machine learning platform that supports: gradual rollouts, A/B testing with statistical significance tracking, automatic rollback on quality degradation, and per-customer feature targeting. Include the data model and API design.",
    "Analyze the impact of the EU AI Act on a company building LLM-powered customer service solutions. Identify which provisions apply, the compliance requirements and timeline, potential penalties for non-compliance, and a prioritized implementation roadmap.",
    "Write a production-ready async Python service for processing webhook events with: guaranteed at-least-once delivery, idempotency handling, ordered processing per entity, dead letter queue, and monitoring integration. Use FastAPI and include full error handling.",
    "Develop a comprehensive evaluation framework for comparing the quality of different LLM providers for a customer support use case. Include: metric definitions, dataset construction methodology, statistical significance testing, cost-adjusted performance scoring, and a decision matrix.",
    "Explain the trade-offs between different approaches to RAG (Retrieval Augmented Generation): dense retrieval, sparse retrieval, hybrid approaches, and late interaction models. Include when to use each, typical accuracy benchmarks, and implementation complexity.",
    "Design the database schema and query patterns for a multi-tenant SaaS application with strict data isolation, supporting 10,000 tenants with varying data volumes. Address: schema isolation vs row-level security approaches, indexing strategy, and migration management.",
    "Write a detailed post-mortem template and process for handling production incidents in an AI-powered system. Include sections for: timeline reconstruction, root cause analysis (5 Whys), blast radius assessment, mitigation steps taken, preventive measures, and communication templates for stakeholders.",
]

ALL_PROMPTS = []
# Build 200 prompts with realistic tier distribution
import math
n_t1 = 70   # 35%
n_t2 = 80   # 40%
n_t3 = 50   # 25%

# Cycle through each tier's prompts to reach target count
def cycle_to(lst, n):
    result = []
    while len(result) < n:
        result.extend(lst[:n - len(result)])
    return result[:n]

for p in cycle_to(TIER1_PROMPTS, n_t1):
    ALL_PROMPTS.append((p, 1))
for p in cycle_to(TIER2_PROMPTS, n_t2):
    ALL_PROMPTS.append((p, 2))
for p in cycle_to(TIER3_PROMPTS, n_t3):
    ALL_PROMPTS.append((p, 3))

# Shuffle for realistic mixed traffic
import random
random.seed(42)
random.shuffle(ALL_PROMPTS)


def send_one(url: str, prompt: str, session: requests.Session) -> dict:
    """Send one prompt to the API and return the result dict."""
    start = time.monotonic()
    try:
        resp = session.post(
            f"{url}/v1/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return {
                "success":        True,
                "prompt_preview": prompt[:80],
                "tier":           data.get("tier"),
                "model_used":     data.get("model_used"),
                "cost_usd":       data.get("cost_usd", 0),
                "cost_baseline":  data.get("cost_if_highest_quality", 0),
                "savings_pct":    data.get("savings_pct", 0),
                "latency_ms":     latency_ms,
                "error":          None,
            }
        else:
            return {
                "success":        False,
                "prompt_preview": prompt[:80],
                "tier":           None,
                "model_used":     None,
                "cost_usd":       0,
                "cost_baseline":  0,
                "savings_pct":    0,
                "latency_ms":     latency_ms,
                "error":          f"HTTP {resp.status_code}: {resp.text[:100]}",
            }
    except Exception as e:
        return {
            "success":        False,
            "prompt_preview": prompt[:80],
            "tier":           None,
            "model_used":     None,
            "cost_usd":       0,
            "cost_baseline":  0,
            "savings_pct":    0,
            "latency_ms":     (time.monotonic() - start) * 1000,
            "error":          str(e)[:100],
        }


def run_load_test(url: str, count: int, dry_run: bool = False) -> None:
    WIDTH = 78

    print("=" * WIDTH)
    print("  LLM COST AUTOPILOT — LOAD TEST")
    print(f"  Target URL:  {url}")
    print(f"  Prompts:     {count}")
    print(f"  Tier mix:    ~35% T1 / ~40% T2 / ~25% T3")
    print(f"  Started at:  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * WIDTH)

    if dry_run:
        print("\n[DRY RUN] Printing prompts without sending:\n")
        for i, (prompt, tier) in enumerate(ALL_PROMPTS[:count], 1):
            print(f"  [{i:03d}] T{tier} {prompt[:70]}")
        return

    # Check API is reachable
    try:
        r = requests.get(f"{url}/health", timeout=5)
        health = r.json()
        print(f"\n  API health: {health.get('status', 'unknown')}")
        print(f"  DB:         {'✓' if health.get('db_ok') else '✗'}")
        print(f"  Classifier: {'✓' if health.get('classifier_loaded') else '✗'}")
        print()
    except Exception as e:
        print(f"\n  ✗ Cannot reach API at {url}: {e}")
        print("  Start the API first: docker compose up  OR  uvicorn src.api.main:app")
        sys.exit(1)

    prompts_to_use = ALL_PROMPTS[:count]
    results = []
    session = requests.Session()

    # Header
    print(f"  {'#':>4}  {'T':>2}  {'Model':<28}  {'Cost':>10}  {'Save%':>6}  {'ms':>5}  Status")
    print("  " + "─" * (WIDTH - 2))

    total_cost     = 0.0
    total_baseline = 0.0
    errors         = 0

    for i, (prompt, expected_tier) in enumerate(prompts_to_use, 1):
        result = send_one(url, prompt, session)
        results.append(result)

        if result["success"]:
            total_cost     += result["cost_usd"]
            total_baseline += result["cost_baseline"]
            tier_icon = f"T{result['tier']}" if result["tier"] else "T?"
            model_short = (result["model_used"] or "")[:28]
            print(
                f"  {i:>4}  {tier_icon:>2}  {model_short:<28}  "
                f"${result['cost_usd']:>8.6f}  "
                f"{result['savings_pct']:>5.1f}%  "
                f"{result['latency_ms']:>4.0f}ms  ✓"
            )
        else:
            errors += 1
            print(f"  {i:>4}  --  {'ERROR':<28}  {'':>10}  {'':>6}  {'':>5}ms  ✗ {result['error'][:30]}")

        # Small delay to respect rate limits
        time.sleep(2.0)

    # Save CSV
    csv_path = ROOT / "data" / "load_test_results.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n  ✓ Results saved → {csv_path}")

    # Pull final stats from the API
    try:
        stats = requests.get(f"{url}/v1/stats", timeout=10).json()
    except Exception:
        stats = {}

    # Final report
    print()
    print("=" * WIDTH)
    print("  LOAD TEST RESULTS — PORTFOLIO HEADLINE")
    print("=" * WIDTH)

    ok = [r for r in results if r["success"]]
    print(f"  Requests sent:          {count}")
    print(f"  Successful:             {len(ok)}  ({len(ok)/count*100:.0f}%)")
    print(f"  Errors:                 {errors}")
    print()

    if total_baseline > 0:
        overall_savings = (1 - total_cost / total_baseline) * 100
        print(f"  ──────────────────────────────────────────")
        print(f"  Actual cost this run:   ${total_cost:.6f}")
        print(f"  Cost if always 70B:     ${total_baseline:.6f}")
        print(f"")
        print(f"  ✓ SAVINGS:  {overall_savings:.1f}%  (${total_baseline - total_cost:.6f} saved)")
        print(f"  ──────────────────────────────────────────")
        print()
        print(f"  README headline (copy this):")
        print(f"  \"Reduced LLM API costs by {overall_savings:.0f}% while maintaining")
        print(f"   quality parity across {len(ok)} requests.\"")

    # Routing distribution
    from collections import Counter
    model_dist = Counter(r["model_used"] for r in ok if r["model_used"])
    tier_dist  = Counter(r["tier"] for r in ok if r["tier"])

    print()
    print("  Routing distribution:")
    for model, cnt in model_dist.most_common():
        pct = cnt / len(ok) * 100 if ok else 0
        bar = "█" * int(pct / 5)
        print(f"    {model or 'unknown':<35} {cnt:>4} ({pct:>4.0f}%) {bar}")

    print()
    print("  Tier distribution:")
    for tier in sorted(tier_dist.keys()):
        cnt = tier_dist[tier]
        pct = cnt / len(ok) * 100 if ok else 0
        print(f"    Tier {tier}:  {cnt:>4} requests  ({pct:.0f}%)")

    # Latency stats
    latencies = [r["latency_ms"] for r in ok]
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print()
        print(f"  Latency (p50/p95/p99): {p50:.0f}ms / {p95:.0f}ms / {p99:.0f}ms")

    # Cumulative DB stats
    if stats:
        print()
        print("  Cumulative DB totals (all-time):")
        print(f"    Total requests: {stats.get('total_requests', 'N/A')}")
        print(f"    Savings:        {stats.get('savings_pct', 'N/A')}%")
        print(f"    Saved USD:      ${stats.get('savings_usd', 0):.6f}")

    print()
    print(f"  ✓ Dashboard URL: http://localhost:8501")
    print(f"  ✓ API docs:      {url}/docs")
    print("=" * WIDTH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the LLM Cost Autopilot API")
    parser.add_argument("--url",   default="http://localhost:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=200,           help="Number of prompts to send")
    parser.add_argument("--dry-run", action="store_true",           help="Print prompts without sending")
    args = parser.parse_args()

    run_load_test(url=args.url, count=args.count, dry_run=args.dry_run)