import sys
from tests.test_embedding import run_embedding_test
from tests.test_rope import run_rope_test
from tests.test_attention import run_attention_test
from tests.test_rmsnorm import run_rmsnorm_test
from tests.test_swiglu import run_swiglu_test
from tests.test_output_projection import run_output_projection_test

print("==================================================")
print("     PHASE 1 CENTRAL VERIFICATION GATEWAY        ")
print("==================================================\n")

try:
    # ==== GATE 1A: TOKEN EMBEDDING ====
    print("Executing Gate 1A: Isolated Token Embedding...")
    emb_error = run_embedding_test()
    print(f"-> [SUCCESS] Max Absolute Error: {emb_error:.2e}")
    print("-> Gate 1A Verification Status: PASSED.\n")
    
    # ==== GATE 1B: ROTARY POSITION EMBEDDINGS ====
    print("Executing Gate 1B: Rotary Position Embeddings (RoPE)...")
    rope_error = run_rope_test()
    print(f"-> [SUCCESS] Max Absolute Error: {rope_error:.2e}")
    print("-> Gate 1B Verification Status: PASSED.\n")
    
    # ==== GATE 1C: MULTI-HEAD ATTENTION ====
    print("Executing Gate 1C: Multi-Head Attention (Causal)...")
    attn_error = run_attention_test()
    print(f"-> [SUCCESS] Max Absolute Error: {attn_error:.2e}")
    print("-> Gate 1C Verification Status: PASSED.\n")
    
    # ==== GATE 1D: ROOT MEAN SQUARE NORMALIZATION ====
    print("Executing Gate 1D: RMSNorm Layer Normalization...")
    rmsnorm_error = run_rmsnorm_test()
    print(f"-> [SUCCESS] Max Absolute Error: {rmsnorm_error:.2e}")
    print("-> Gate 1D Verification Status: PASSED.\n")
    
    # ==== GATE 1E: SWIGLU FEED-FORWARD ====
    print("Executing Gate 1E: SwiGLU Gated Feed-Forward...")
    swiglu_error = run_swiglu_test()
    print(f"-> [SUCCESS] Max Absolute Error: {swiglu_error:.2e}")
    print("-> Gate 1E Verification Status: PASSED.\n")
    
    # ==== GATE 1F: OUTPUT PROJECTION ====
    print("Executing Gate 1F: Output Projection (Logits)...")
    proj_error = run_output_projection_test()
    print(f"-> [SUCCESS] Max Absolute Error: {proj_error:.2e}")
    print("-> Gate 1F Verification Status: PASSED.\n")
    
    # ==== SUMMARY ====
    print("==================================================")
    print("STATUS: All Phase 1 Components (1A-1F) Verified")
    print("==================================================")
    print(f"1A Embedding Error:      {emb_error:.2e} (<1e-4 ✓)")
    print(f"1B RoPE Error:           {rope_error:.2e} (<1e-4 ✓)")
    print(f"1C Attention Error:      {attn_error:.2e} (<1e-4 ✓)")
    print(f"1D RMSNorm Error:        {rmsnorm_error:.2e} (<1e-4 ✓)")
    print(f"1E SwiGLU Error:         {swiglu_error:.2e} (<1e-4 ✓)")
    print(f"1F Output Projection:    {proj_error:.2e} (<1e-4 ✓)")
    print("==================================================\n")
    print("PHASE 1 COMPLETE: All forward components verified.")
    print("Ready for Phase 2: Backward pass & training loop.\n")
    
except AssertionError as e:
    print(f"\n[CRITICAL FAILURE] Verification Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[UNEXPECTED EXCEPTION] Run halted: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)