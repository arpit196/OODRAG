"""Explicit offline CI suite. Real-corpus/model tests belong to integration runs."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PATTERNS = [
    'test_assessment_output_guard.py', 'test_assessment_source_spans.py',
    'test_batched_citations.py', 'test_customer_policy_record.py',
    'test_evidence_gate.py', 'test_evidence_quote_whitespace.py',
    'test_expansion_budget.py', 'test_requested_policy_scope.py',
    'test_simple_insurance_results.py', 'test_insurance_reporting_ui.py',
    'test_boundary_direct_query.py',
]


def main():
    missing = [name for name in PATTERNS if not (ROOT / 'tests' / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing required CI modules: {missing}')
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=p)
                               for p in PATTERNS)
    expected = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {'suite': 'offline_ci', 'modules': PATTERNS, 'discovered': expected,
               'run': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'skipped': len(result.skipped),
               'expected_failures': len(result.expectedFailures),
               'unexpected_successes': len(result.unexpectedSuccesses),
               'scope': 'Mocked contracts and arithmetic; no real-corpus recall or LLM quality claims.'}
    output = ROOT / 'ci-results'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Missing/empty modules and unexpected skips must not silently make CI green.
    return int(not result.wasSuccessful() or expected == 0 or result.testsRun != expected
               or bool(result.skipped) or bool(result.expectedFailures))


if __name__ == '__main__':
    raise SystemExit(main())
