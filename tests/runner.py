"""Run a test module's test_* functions without pytest."""


def run_tests(namespace, title):
    tests = [v for k, v in sorted(namespace.items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(tests)} {title} TESTS PASSED")
