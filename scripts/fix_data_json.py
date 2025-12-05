import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    fix_count = 0
    for record in data:
        if "<image>" not in record["conversations"][0]["value"]:
            record["conversations"][0]["value"] = "<image>\n" + record["conversations"][0]["value"]
            fix_count += 1

    print(f"Fixed {fix_count} records")

    with open(args.json_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()