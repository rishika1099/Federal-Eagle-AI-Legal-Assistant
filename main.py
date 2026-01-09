# main.py
from dotenv import load_dotenv
from crew import legal_assistant_crew
import sys

load_dotenv()


def run(user_input: str, case_name: str = "Legal Case"):
    """Run legal analysis on the given input."""
    print("\n")
    print(f"ANALYZING: {case_name.upper()}")
    print("\n")
    print(f"Scenario: {user_input}")
    print("\n")
    
    try:
        result = legal_assistant_crew.kickoff(inputs={"user_input": user_input})
        
        print("\n")
        print("ANALYSIS COMPLETE")
        print("\n")
        print(result)
        print("\n")
        
        return result
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return None


# Test cases
TEST_CASES = {
    "computer_fraud": {
        "name": "Computer Fraud and Espionage",
        "scenario": (
            "An individual accessed a protected government computer system without "
            "authorization using stolen credentials. They downloaded classified documents "
            "relating to national defense and attempted to transmit them via email to "
            "a foreign intelligence contact. The unauthorized access occurred over a "
            "two-week period from Virginia, and the attempted transmission was intercepted "
            "by federal authorities."
        )
    },
    "wire_fraud": {
        "name": "Wire Fraud Investment Scheme",
        "scenario": (
            "A person created a fake investment website and sent promotional emails to "
            "potential investors across multiple states. They used interstate wire "
            "communications to solicit over $500,000 from victims, promising guaranteed "
            "20 percent returns on a cryptocurrency fund that did not exist. The scheme "
            "operated for six months before victims reported losses to the FBI."
        )
    },
    "bank_robbery": {
        "name": "Bank Robbery Federal Property",
        "scenario": (
            "Two individuals entered a federally-insured bank with firearms, threatened "
            "employees and customers, and stole approximately $75,000 in cash. They fled "
            "across state lines from California to Nevada and were apprehended three days "
            "later during a traffic stop."
        )
    },
    "identity_theft": {
        "name": "Identity Theft Interstate Commerce",
        "scenario": (
            "Someone used my stolen Social Security number and personal information to "
            "open multiple credit card accounts and bank accounts in my name. They made "
            "over $50,000 in fraudulent purchases across several states using online "
            "shopping sites. I discovered this when I was denied a loan due to damaged credit."
        )
    },
    "drug_trafficking": {
        "name": "Drug Trafficking Interstate",
        "scenario": (
            "An individual was caught transporting 50 kilograms of cocaine from Texas to "
            "New York using commercial interstate highways. They were stopped at a border "
            "checkpoint and the drugs were discovered hidden in vehicle compartments. "
            "This is their first offense."
        )
    },
    "money_laundering": {
        "name": "Money Laundering Financial Institution",
        "scenario": (
            "A person established multiple shell companies and used them to process over "
            "$2 million in proceeds from illegal gambling operations. They structured "
            "deposits to avoid currency reporting requirements and transferred funds "
            "through multiple banks across different states to conceal the source."
        )
    },
    "kidnapping": {
        "name": "Interstate Kidnapping",
        "scenario": (
            "An individual abducted a child from a school in Maryland and transported "
            "the victim across state lines to Pennsylvania. A ransom demand of $100,000 "
            "was sent to the parents via email. The child was recovered unharmed after "
            "three days when the suspect was located by federal authorities."
        )
    },
    "tax_evasion": {
        "name": "Tax Evasion and Fraud",
        "scenario": (
            "A business owner failed to report over $800,000 in income over five years "
            "and created false invoices and expense records to claim fraudulent deductions. "
            "They also failed to pay employment taxes for 15 employees and kept two sets "
            "of books to hide income from the IRS."
        )
    }
}


def main():
    """Main entry point with test case selection."""
    
    if len(sys.argv) > 1:
        # Run specific test case from command line
        case_key = sys.argv[1].lower()
        if case_key in TEST_CASES:
            test_case = TEST_CASES[case_key]
            run(test_case["scenario"], test_case["name"])
        elif sys.argv[1] == "all":
            # Run all test cases
            print("\nRunning all test cases...\n")
            for case_key, test_case in TEST_CASES.items():
                print(f"\n\nTest Case: {case_key}")
                run(test_case["scenario"], test_case["name"])
                print("\n" + "Next case..." + "\n")
        else:
            print(f"Unknown test case: {case_key}")
            print(f"Available cases: {', '.join(TEST_CASES.keys())}")
            print("Or use 'all' to run all test cases")
    else:
        # Run default test case
        default_case = TEST_CASES["computer_fraud"]
        print("Running default test case: Computer Fraud and Espionage")
        print("To run specific cases, use: python main.py <case_name>")
        print(f"Available: {', '.join(TEST_CASES.keys())}")
        print("Or use 'python main.py all' to run all test cases")
        print("\n")
        
        run(default_case["scenario"], default_case["name"])


if __name__ == "__main__":
    main()