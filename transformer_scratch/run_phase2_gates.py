import sys
from tests.test_gradient_check import run_gradient_check_test
from tests.test_overfit import run_overfit_test
from tests.test_training_loop import run_training_loop_test

print("==================================================")
print("     PHASE 2 CENTRAL VERIFICATION GATEWAY        ")
print("==================================================\n")

try:
    # ==== GATE 2A: GRADIENT CHECKS ====
    print("Executing Gate 2A: Gradient Checks (Finite Difference)...")
    print()
    grad_results = run_gradient_check_test(num_checks=5)
    print()
    print("-> Gate 2A Verification Status: PASSED.\n")
    
    # ==== GATE 2B: OVERFIT TEST ====
    print("Executing Gate 2B: Overfit Single Batch...")
    print()
    initial_loss, final_loss = run_overfit_test(num_iterations=100)
    print()
    print("-> Gate 2B Verification Status: PASSED.\n")
    
    # ==== GATE 2C: TRAINING LOOP ====
    print("Executing Gate 2C: Full Training Loop...")
    print()
    results = run_training_loop_test(num_epochs=2, batch_size=4)
    print()
    print("-> Gate 2C Verification Status: PASSED.\n")
    
    # ==== SUMMARY ====
    print("==================================================")
    print("STATUS: All Phase 2 Components (2A, 2B, 2C) Verified")
    print("==================================================")
    print(f"2A Gradient Checks:  ✅ PASSED (< 1e-4 tolerance)")
    print(f"2B Overfit Test:     ✅ PASSED ({initial_loss:.4f} → {final_loss:.6f})")
    print(f"2C Training Loop:    ✅ PASSED (Loss decreased)")
    print("==================================================\n")
    print("PHASE 2 COMPLETE: Backward pass & training verified.")
    print("Ready for Phase 3: Train to convergence on real data.\n")
    
except AssertionError as e:
    print(f"\n[CRITICAL FAILURE] Verification Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[UNEXPECTED EXCEPTION] Run halted: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)