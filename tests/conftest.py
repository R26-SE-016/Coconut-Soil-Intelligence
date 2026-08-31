def pytest_terminal_summary(terminalreporter, exitstatus, config):
    stats = terminalreporter.stats
    passed_reports = stats.get('passed', [])
    failed_reports = stats.get('failed', [])
    
    unit_total = 0
    unit_passed = 0
    unit_failed = 0
    
    int_total = 0
    int_passed = 0
    int_failed = 0
    
    for rep in passed_reports + failed_reports:
        nodeid = rep.nodeid
        is_passed = (rep.outcome == 'passed')
        
        # Categorize by filename
        if 'test_cri_recommendations' in nodeid or 'test_validation' in nodeid:
            unit_total += 1
            if is_passed:
                unit_passed += 1
            else:
                unit_failed += 1
        elif 'test_api_integration' in nodeid or 'test_image_predict' in nodeid:
            int_total += 1
            if is_passed:
                int_passed += 1
            else:
                int_failed += 1

    terminalreporter.write_sep("=", "SARUPOL TEST SUITE CATEGORY BREAKDOWN")
    terminalreporter.write_line(f" 📌 UNIT TESTS:        Total = {unit_total}  | Passed = {unit_passed}  | Failed = {unit_failed}")
    terminalreporter.write_line(f" 📌 INTEGRATION TESTS: Total = {int_total}   | Passed = {int_passed}   | Failed = {int_failed}")
    terminalreporter.write_line(f" ------------------------------------------------------------------")
    terminalreporter.write_line(f" 🚀 OVERALL RESULTS:   Total = {unit_total + int_total}  | Passed = {unit_passed + int_passed}  | Failed = {unit_failed + int_failed}")
    terminalreporter.write_sep("=")
