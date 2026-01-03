"""
Demo Script - Text Summarizer
==============================
Showcases the text summarization capabilities with example dialogues.
Run with: python demo.py
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.text_summarizer.pipeline.inference import Summarizer


# Example dialogues for demonstration
EXAMPLE_DIALOGUES = [
    {
        "name": "Party Invitation",
        "dialogue": """John: Hey, are you coming to the party tonight?
Sarah: I'm not sure, I have a lot of work to do.
John: Come on, it'll be fun! Everyone's going to be there.
Sarah: Okay, I'll try to come by 8.
John: Great! See you then!"""
    },
    {
        "name": "Meeting Scheduling",
        "dialogue": """Alice: Hi Bob, can we schedule a meeting for tomorrow?
Bob: Sure, what time works for you?
Alice: How about 2 PM?
Bob: That works. Should I book the conference room?
Alice: Yes please. We need to discuss the Q4 budget.
Bob: Got it. I'll send the calendar invite."""
    },
    {
        "name": "Restaurant Recommendation",
        "dialogue": """Emma: I'm looking for a good Italian restaurant. Any suggestions?
Mike: Have you tried Bella Italia on Main Street?
Emma: No, is it good?
Mike: Amazing! Their pasta is homemade and the tiramisu is to die for.
Emma: Sounds perfect. Is it expensive?
Mike: Moderate pricing. About $25-30 per person.
Emma: Great, I'll make a reservation for Friday."""
    },
    {
        "name": "Technical Support",
        "dialogue": """Customer: My laptop won't turn on. I've tried everything.
Support: Have you tried holding the power button for 10 seconds?
Customer: Yes, nothing happens.
Support: Is the charging light on when you plug it in?
Customer: No, there's no light at all.
Support: It sounds like a power adapter issue. Try a different charger if you have one.
Customer: I don't have another one.
Support: I'll arrange for a replacement adapter to be sent to you."""
    },
    {
        "name": "Travel Planning",
        "dialogue": """Tom: I'm thinking of going to Japan next spring.
Lisa: Oh, that's a great time! Cherry blossom season.
Tom: Exactly! Any tips?
Lisa: Book hotels early, they fill up fast during hanami.
Tom: How long should I stay?
Lisa: At least two weeks to see Tokyo, Kyoto, and Osaka.
Tom: That sounds perfect. I'll start planning!"""
    },
    {
        "name": "Project Update",
        "dialogue": """Manager: How's the project coming along?
Developer: We're about 80% done. The main features are complete.
Manager: What's left?
Developer: Testing and documentation. Should take about a week.
Manager: Any blockers?
Developer: We need access to the production database for final testing.
Manager: I'll get that sorted today.
Developer: Thanks! We should be ready for deployment by Friday."""
    },
    {
        "name": "Birthday Surprise",
        "dialogue": """Amy: Don't forget, Jake's surprise party is Saturday at 7.
Ben: I remember! Should I bring anything?
Amy: Can you get the cake from Sweet Delights?
Ben: Sure, chocolate or vanilla?
Amy: Chocolate, it's his favorite.
Ben: Got it. How many people are coming?
Amy: About 15. I've decorated the backyard already.
Ben: Perfect, see you Saturday!"""
    },
    {
        "name": "Gym Membership",
        "dialogue": """Staff: Welcome to FitLife Gym! How can I help you?
Customer: I'd like to know about membership options.
Staff: We have monthly at $50 or annual at $450.
Customer: Does that include classes?
Staff: Yes, all group classes are included. Personal training is extra.
Customer: Can I try it first?
Staff: Absolutely! We offer a free 3-day trial.
Customer: Great, I'll start with that."""
    },
]


def main():
    print("=" * 60)
    print("  TEXT SUMMARIZER - DEMO")
    print("  Using: philschmid/flan-t5-base-samsum")
    print("=" * 60)
    
    # Initialize summarizer
    print("\nLoading model...")
    start_load = time.time()
    summarizer = Summarizer()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds\n")
    
    total_inference_time = 0
    results = []
    
    for i, example in enumerate(EXAMPLE_DIALOGUES, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {example['name']}")
        print("=" * 60)
        
        print("\n📝 DIALOGUE:")
        print("-" * 40)
        print(example['dialogue'])
        
        # Generate summary
        start_time = time.time()
        summary = summarizer.summarize(example['dialogue'])
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        print("\n✨ SUMMARY:")
        print("-" * 40)
        print(summary)
        print(f"\n⏱️  Inference time: {inference_time:.2f}s")
        
        results.append({
            "name": example['name'],
            "summary": summary,
            "time": inference_time
        })
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\n  Model load time: {load_time:.2f}s")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average per summary: {total_inference_time/len(EXAMPLE_DIALOGUES):.2f}s")
    print(f"  Examples processed: {len(EXAMPLE_DIALOGUES)}")
    
    print("\n" + "=" * 60)
    print("  ALL SUMMARIES")
    print("=" * 60)
    for r in results:
        print(f"\n  [{r['name']}]")
        print(f"  {r['summary']}")
    
    print("\n✅ Demo complete!")
    return results


if __name__ == "__main__":
    main()
