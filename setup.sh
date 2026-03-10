#!/usr/bin/env bash
# ============================================================
# phi-Enhanced RLM -- One-Click Setup
# ============================================================
# Run: bash setup.sh
# That's it. This handles everything.
# ============================================================
set -e

echo ""
echo "========================================================"
echo "  phi-Enhanced Recursive Language Model -- Setup"
echo "  Version 4.0.0"
echo "========================================================"
echo ""

# Step 1: Check Python
echo "[1/5] Checking Python..."
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "  ERROR: Python 3 not found."
    echo "  Please install Python 3.9+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo "  ERROR: Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "  OK: Python $PYTHON_VERSION found"

# Step 2: Install the package
echo ""
echo "[2/5] Installing phi-RLM and dependencies..."
if pip install -e "." --quiet 2>/dev/null; then
    echo "  OK: Core package installed"
elif pip install -e "." --break-system-packages --quiet 2>/dev/null; then
    echo "  OK: Core package installed"
else
    echo "  ERROR: pip install failed. Try running manually:"
    echo "    pip install -e ."
    exit 1
fi

# Step 3: Create .env if it doesn't exist
echo ""
echo "[3/5] Setting up configuration..."
if [ ! -f .env ]; then
    if [ -f .env.template ]; then
        cp .env.template .env
        echo "  OK: Created .env from template"
    else
        echo "  SKIP: No .env.template found"
    fi
    echo ""
    echo "  NOTE: For real AI responses (not mock), edit .env and add"
    echo "  your OpenRouter API key. Get one at: https://openrouter.ai/keys"
    echo "  Without a key, the system runs in demo mode using mock"
    echo "  responses -- still fully functional for testing."
else
    echo "  OK: .env already exists"
fi

# Step 4: Smoke test
echo ""
echo "[4/5] Verifying installation..."
$PYTHON -c "
from src import PhiEnhancedRLM, PHI, PhiEvolutionEngine, MetaRecursiveRLM
from src.agent_router import AgentRouter
from src.skill_loader import SkillLoader
from src.phi_enhanced_rlm import MockLLMBackend

# Quick smoke test
mock = MockLLMBackend(seed=42)
rlm = PhiEnhancedRLM(
    base_llm_callable=mock,
    context_chunks=['Test chunk for verification.'],
    total_budget_tokens=512
)
result = rlm.recursive_solve('Verify installation', max_depth=2)
assert result.confidence > 0, 'Confidence should be positive'
router = AgentRouter()
assert len(router.get_available_agents()) >= 7, 'Should have 7+ agent personas'
print(f'  OK: Core engine works (confidence: {result.confidence:.1%})')
print(f'  OK: Golden ratio phi = {PHI}')
print(f'  OK: Agent router loaded {len(router.get_available_agents())} agents')
print(f'  OK: All modules imported successfully')
" 2>&1 | grep -E "^\s+OK:"

# Step 5: Show next steps
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "========================================================"
echo "  phi-RLM is ready to use!"
echo "========================================================"
echo ""
echo "  Try these commands:"
echo ""
echo "    python quickstart.py              # See it work in 10 seconds"
echo "    python -m src.evolution_loop      # Watch it self-evolve"
echo "    python -m pytest tests/ -v        # Run all tests"
echo ""
echo "  For the full guide, see README.md"
echo ""
