"""
run.py — SIS: Satellite Intelligence System
Ask a natural language question, get a direct answer from orbit.
"""

import argparse
import json
from pipeline import SISPipeline

EXAMPLE_QUESTIONS = [
    "Is there flooding near the coast?",
    "Which areas have crop stress and need attention?",
    "Show me healthy forested regions",
    "Where is urban expansion happening?",
    "Are there any water bodies detected?",
]


def print_answer(output: dict):
    print(f"\n{'='*60}")
    print(f"❓ QUESTION : {output['question']}")
    print(f"🛰️  INTENT   : {output['intent_summary']}")
    print(f"{'='*60}")
    print(f"\n{output['answer']}\n")

    if output["matched_tiles"]:
        print("📍 Locations:")

        for t in output["matched_tiles"][:5]:
            print(f"   • {t['region']} "
            f"— conf={t['confidence']:.2f} "
            f"NDVI={t['ndvi']:.2f} "
            f"water={t['water']:.2f}")

    m = output["meta"]
    print(f"\n📡 DOWNLINK STATS:")
    print(f"   Tiles scanned     : {m['total_tiles']}")
    print(f"   Matches found     : {m['tiles_matched']}")
    print(f"   Raw imagery would : {m['raw_imagery_mb']} MB")
    print(f"   Downlink size     : {m['downlink_bytes']} bytes")
    print(f"   Bandwidth saved   : {m['bandwidth_reduction']}")
    print(f"   Latency           : {m['latency_ms']}ms")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SIS — Satellite Intelligence System\n"
                    "Ask a question, get a direct answer from orbit."
    )
    parser.add_argument("--tiles",    type=int,  default=60)
    parser.add_argument("--question", type=str,  default=None)
    parser.add_argument("--demo",     action="store_true")
    args = parser.parse_args()

    print("🛰️  Initializing Satellite Intelligence System...")
    pipeline = SISPipeline(n_tiles=args.tiles)

    if args.demo:
        for q in EXAMPLE_QUESTIONS:
            print_answer(pipeline.ask(q))
        return

    if args.question:
        print_answer(pipeline.ask(args.question))
        return

    # Interactive mode
    print("\n🛰️  SIS ready. Ask a question about what the satellite sees.\n")
    print("Example questions:")
    for q in EXAMPLE_QUESTIONS:
        print(f"  • {q}")
    print()

    while True:
        try:
            question = input("ask> ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break
            print_answer(pipeline.ask(question))
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()