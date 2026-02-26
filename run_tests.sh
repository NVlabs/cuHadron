# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash

# cuHadron - Test Runner  
# Usage: ./run_tests.sh [--baseline] [--fixed] [all|<category>|<test_name>]
# Note: Tests are always rebuilt by default to ensure consistency

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Arrays to track test results
declare -a TESTS_WITH_RACES
declare -a TESTS_WITHOUT_RACES
declare -a TESTS_UNKNOWN
declare -a TESTS_KILLED
declare -a FALSE_POSITIVE_TESTS_CORRECT
declare -a FALSE_POSITIVE_TESTS_FAILED

# Optional: run each test under your race detector. If set, the script runs
#   timeout 600 $RUN_CMD "$test_path"
# instead of the binary directly. Your tool must exit 0 = no race, 1 = race detected.
# Example: RUN_CMD="compute-sanitizer --tool=racecheck" ./run_tests.sh all
RUN_CMD="${RUN_CMD:-}"

# False positive check tests (should NOT detect races - any detection is a false positive)
# These tests access different bytes in the same word - should be handled correctly
FALSE_POSITIVE_CHECK_TESTS=(
    "false_positives/shared_bytewise_readwrite"
    "false_positives/shared_bytewise_writewrite"
    "false_positives/global_bytewise_readwrite"
    "false_positives/global_bytewise_writewrite"
)

# Category directories (must match test_requirements.txt and directory layout)
TEST_CATEGORIES=(memcpy bulkcpy interkernel hostdevice asyncmemcpy intersubwarp dsmem multigpu false_positives interwave)
CATEGORY_PATTERN="memcpy|bulkcpy|interkernel|hostdevice|asyncmemcpy|intersubwarp|dsmem|multigpu|false_positives|interwave"

# Parse flags
BASELINE=0
FIXED=0
REBUILD=1  # Always rebuild by default

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            BASELINE=1
            shift
            ;;
        --fixed)
            FIXED=1
            shift
            ;;
        --rebuild)
            REBUILD=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Architecture: build target and binary suffix.
# User can set ARCH (e.g. ARCH=sm_80, sm_90, sm_100); else we detect from nvidia-smi.
detect_arch() {
    local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ "$compute_cap" =~ ^[0-9]+$ ]]; then
        if [ "$compute_cap" -ge 100 ]; then
            echo "sm100"
        elif [ "$compute_cap" -ge 90 ]; then
            echo "sm90"
        elif [ "$compute_cap" -ge 80 ]; then
            echo "sm80"
        else
            echo "sm80"
        fi
    else
        echo "sm80"
    fi
}

DETECTED_ARCH=$(detect_arch)
# ARCH from env (e.g. sm_90, sm_100) overrides; suffix has no underscore for filenames
if [ -n "${ARCH:-}" ]; then
    BINARY_SUFFIX="${ARCH//_/}"
    echo -e "${CYAN}Using ARCH=$ARCH (binaries *.$BINARY_SUFFIX.out)${NC}"
else
    BINARY_SUFFIX="$DETECTED_ARCH"
    ARCH="sm_${DETECTED_ARCH#sm}"   # sm90 -> sm_90 for make
    echo -e "${CYAN}Detected GPU: $DETECTED_ARCH (binaries *.$BINARY_SUFFIX.out)${NC}"
fi

# Display mode and build configuration
if [ $BASELINE -eq 1 ]; then
    echo -e "${CYAN}Mode: Baseline (no race detection)${NC}"
else
    echo -e "${CYAN}Mode: Run tests (plain binaries)${NC}"
fi

if [ $FIXED -eq 1 ]; then
    echo -e "${CYAN}Building with FIXED=1 (races fixed)${NC}"
else
    echo -e "${CYAN}Building with FIXED=0 (races injected)${NC}"
fi

# Parse target first to determine what to build
TARGET="${1:-all}"

# Build only what's needed based on target
echo -e "${YELLOW}Building tests...${NC}"

case "$TARGET" in
    all)
        if [ $REBUILD -eq 1 ] || ! ls "${TEST_CATEGORIES[0]}"/*.$BINARY_SUFFIX.out >/dev/null 2>&1; then
            echo "Building all tests (clean build)..."
            if [ $REBUILD -eq 1 ]; then
                echo "Rebuild requested - cleaning $BINARY_SUFFIX binaries..."
                make clean ARCH=$ARCH
            fi
            echo "Rebuilding for ARCH=$ARCH..."
            make FIXED=$FIXED ARCH=$ARCH
        else
            echo "All tests already built for $BINARY_SUFFIX"
        fi
        ;;
    $CATEGORY_PATTERN)
        echo "Building $TARGET tests..."
        # Clean only the target directory binaries if rebuild requested
        if [ $REBUILD -eq 1 ]; then
            echo "Cleaning $TARGET binaries..."
            rm -f "$TARGET"/*.$BINARY_SUFFIX.out
        fi
        # Find all .cu files in the directory and build their corresponding .out files
        cu_files=("$TARGET"/*.cu)
        if [ -e "${cu_files[0]}" ]; then
            for cu_file in "${cu_files[@]}"; do
                test_base="${cu_file%.cu}"
                target_binary="${test_base}.$BINARY_SUFFIX.out"
                if [ $REBUILD -eq 1 ] || [ ! -f "$target_binary" ]; then
                    make FIXED=$FIXED ARCH=$ARCH "$target_binary"
                fi
            done
        else
            echo "No tests found in $TARGET/"
        fi
        ;;
    *)
        # Specific test file
        test_base="${TARGET%.out}"
        test_base="${test_base%.cu}"
        if [ -f "${test_base}.cu" ]; then
            echo "Building $test_base..."
            target_binary="${test_base}.$BINARY_SUFFIX.out"
            if [ $REBUILD -eq 1 ] || [ ! -f "$target_binary" ]; then
                # Clean only this specific test if rebuild requested
                if [ $REBUILD -eq 1 ]; then
                    echo "Cleaning ${test_base} binaries..."
                    rm -f "${test_base}.$BINARY_SUFFIX.out"
                fi
                make FIXED=$FIXED ARCH=$ARCH "$target_binary"
            fi
        else
            # Will be caught later with proper error message
            make clean ARCH=$ARCH
            make FIXED=$FIXED ARCH=$ARCH
        fi
        ;;
esac

# Function to run a single test (with architecture-specific binary)
run_test() {
    local test_base="$1"  # e.g., "memcpy/shared_writewrite_race" or "memcpy/shared_writewrite_race.sm_80.out"
    
    # If already an architecture-specific binary, use as-is; otherwise construct it
    local test_path
    if [[ "$test_base" == *".sm"[0-9]*".out" ]]; then
        test_path="$test_base"
    else
        test_path="${test_base%.out}.$BINARY_SUFFIX.out"
    fi
    
    if [ ! -f "$test_path" ]; then
        echo -e "${YELLOW}⊘ Skipping $test_base (no binary for $BINARY_SUFFIX)${NC}\n"
        return 0
    fi
    
    echo -e "${BLUE}=== Running: $test_path ===${NC}"
    
    # Capture output to a temporary file
    local tmpfile=$(mktemp)
    
    # Run test (optionally under RUN_CMD, e.g. a race detector). Timeout 600s.
    # Pass RUN_CMD and test_path separately so RUN_CMD is executed as command + args, with path as last arg.
    if [ -n "$RUN_CMD" ]; then
        timeout 600 bash -c 'eval "$1 \"$2\""' _ "$RUN_CMD" "$test_path" 2>&1 | tee "$tmpfile"
    else
        timeout 600 "$test_path" 2>&1 | tee "$tmpfile"
    fi
    # Use exit code of first command in pipeline (detector/test), not tee
    local exit_code=${PIPESTATUS[0]}
    
    # Check for timeout (exit code 124)
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}✗ Completed - KILLED (timeout after 600 seconds)${NC}\n"
        TESTS_KILLED+=("$test_path")
        rm -f "$tmpfile"
        return 0
    fi
    
    # Check if this is a false positive check test (should NOT detect races)
    local is_false_positive_check=0
    for fp_prefix in "${FALSE_POSITIVE_CHECK_TESTS[@]}"; do
        if [[ "$test_path" == *"$fp_prefix"* ]]; then
            is_false_positive_check=1
            break
        fi
    done
    
    # Classify by exit code only: 0 = no race, 1 = race detected, other = unknown
    if [ $exit_code -eq 0 ]; then
        if [ $is_false_positive_check -eq 1 ]; then
            echo -e "${GREEN}✓ Completed - Correct (no false positive)${NC}\n"
            FALSE_POSITIVE_TESTS_CORRECT+=("$test_path")
        else
            echo -e "${YELLOW}✓ Completed - No race detected${NC}\n"
            TESTS_WITHOUT_RACES+=("$test_path")
        fi
    elif [ $exit_code -eq 1 ]; then
        if [ $is_false_positive_check -eq 1 ]; then
            echo -e "${RED}✗ Completed - FALSE POSITIVE (test should NOT detect races)${NC}\n"
            FALSE_POSITIVE_TESTS_FAILED+=("$test_path")
        else
            echo -e "${GREEN}✓ Completed - Race detected${NC}\n"
            TESTS_WITH_RACES+=("$test_path")
        fi
    else
        echo -e "${GREEN}✓ Completed (exit $exit_code)${NC}\n"
        TESTS_UNKNOWN+=("$test_path")
    fi

    # Clean up
    rm -f "$tmpfile"
    
    if [ $exit_code -ne 0 ]; then
        echo -e "${YELLOW}⚠ Exit code: $exit_code${NC}\n"
    fi
    
    return 0
}

# Function to run tests in a directory
run_category() {
    local category="$1"
    local tests=("$category"/*.$BINARY_SUFFIX.out)
    
    for test in "${tests[@]}"; do
        if [ -f "$test" ]; then
            run_test "$test"
        fi
    done
}

# Run tests
echo -e "${YELLOW}Running tests...${NC}"

case "$TARGET" in
    all)
        # Run all available tests
        for dir in "${TEST_CATEGORIES[@]}"; do
            if [ -d "$dir" ]; then
                run_category "$dir"
            fi
        done
        ;;
    
    $CATEGORY_PATTERN)
        run_category "$TARGET"
        ;;
    
    *)
        # Specific test (strip .cu or .out if provided)
        test_base="${TARGET%.out}"
        test_base="${test_base%.cu}"
        
        if [ -f "${test_base}.cu" ]; then
            run_test "$test_base"
        else
            echo -e "${RED}Error: Test not found: $TARGET${NC}"
            echo ""
            echo "Usage: $0 [--baseline] [--fixed] [--rebuild] [all|<category>|<test>]"
            echo "Categories: ${TEST_CATEGORIES[*]}"
            echo ""
            echo "Flags:"
            echo "  --baseline          Run tests (no detector)"
            echo "  --fixed             Build with FIXED=1 (races fixed, no races expected)"
            echo "  --rebuild           Force rebuild of target tests"
            echo ""
            echo "Examples:"
            echo "  $0 all                                    # Run all tests"
            echo "  $0 --baseline all                         # Run tests (no detector)"
            echo "  $0 --fixed all                            # Run fixed version (no races)"
            echo "  $0 --rebuild memcpy                       # Rebuild and run memcpy tests"
            echo "  $0 memcpy/shared_writewrite_race          # Run specific test"
            echo ""
            echo "Note: Binaries are *.$BINARY_SUFFIX.out (set ARCH=sm_80, sm_90, sm_100 etc.; see test_requirements.txt)"
            echo "      Current: ARCH=$ARCH"
            exit 1
        fi
        ;;
esac

# Print summary report
total_tests=$((${#TESTS_WITH_RACES[@]} + ${#TESTS_WITHOUT_RACES[@]} + ${#TESTS_UNKNOWN[@]} + ${#TESTS_KILLED[@]}))
if [ $total_tests -gt 1 ]; then
    if [ $BASELINE -eq 1 ]; then
        # Baseline mode: just ran binaries, no detector
        echo ""
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}             BASELINE RUN SUMMARY                        ${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "Total tests run: $total_tests"
        echo -e "Mode: No race detector (binaries ran directly)"
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo ""
    elif [ $FIXED -eq 1 ]; then
        # Fixed (race-free) run: no race detected = correct (✓), race detected = false positive (✗)
        echo ""
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}             FIXED RUN SUMMARY (race-free code)          ${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "Expected: No races (code is fixed). Detector should NOT report races."
        echo ""
        tests_without_races_regular=()
        for test in "${TESTS_WITHOUT_RACES[@]}"; do
            is_fp_check=0
            for fp_test in "${FALSE_POSITIVE_CHECK_TESTS[@]}"; do
                if [[ "$test" == *"$fp_test"* ]]; then
                    is_fp_check=1
                    break
                fi
            done
            [ $is_fp_check -eq 0 ] && tests_without_races_regular+=("$test")
        done
        if [ ${#tests_without_races_regular[@]} -gt 0 ]; then
            echo -e "${GREEN}✓ Correct - No race detected (${#tests_without_races_regular[@]}):${NC}"
            for test in "${tests_without_races_regular[@]}"; do
                echo -e "  ${GREEN}✓${NC} $test"
            done
            echo ""
        fi
        if [ ${#TESTS_WITH_RACES[@]} -gt 0 ]; then
            echo -e "${RED}✗ False positive - Race incorrectly detected (${#TESTS_WITH_RACES[@]}):${NC}"
            for test in "${TESTS_WITH_RACES[@]}"; do
                echo -e "  ${RED}✗${NC} $test"
            done
            echo ""
        fi
        if [ ${#FALSE_POSITIVE_TESTS_CORRECT[@]} -gt 0 ]; then
            echo -e "${GREEN}✓ False-positive checks - Correct (${#FALSE_POSITIVE_TESTS_CORRECT[@]}):${NC}"
            for test in "${FALSE_POSITIVE_TESTS_CORRECT[@]}"; do
                echo -e "  ${GREEN}✓${NC} $test"
            done
            echo ""
        fi
        echo -e "${CYAN}────────────────────────────────────────────────────────${NC}"
        echo -e "Total tests run: $total_tests"
        echo -e "Correct (no race): ${GREEN}$((${#tests_without_races_regular[@]} + ${#FALSE_POSITIVE_TESTS_CORRECT[@]}))${NC}"
        [ ${#TESTS_WITH_RACES[@]} -gt 0 ] && echo -e "False positives: ${RED}${#TESTS_WITH_RACES[@]}${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo ""
    else
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}             RACE DETECTION REPORT (plain run)            ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Tests with races detected
    if [ ${#TESTS_WITH_RACES[@]} -gt 0 ]; then
        echo -e "${GREEN}✓ Tests with races detected (${#TESTS_WITH_RACES[@]}):${NC}"
        for test in "${TESTS_WITH_RACES[@]}"; do
            echo -e "  ${GREEN}✓${NC} $test"
        done
        echo ""
    fi
    
    # Tests without races detected (excluding false positive checks)
    tests_without_races_regular=()
    for test in "${TESTS_WITHOUT_RACES[@]}"; do
        # Check if it's a false positive check test
        is_fp_check=0
        for fp_test in "${FALSE_POSITIVE_CHECK_TESTS[@]}"; do
            if [[ "$test" == *"$fp_test"* ]]; then
                is_fp_check=1
                break
            fi
        done
        
        if [ $is_fp_check -eq 0 ]; then
            tests_without_races_regular+=("$test")
        fi
    done
    
    if [ ${#tests_without_races_regular[@]} -gt 0 ]; then
        echo -e "${YELLOW}✗ Tests with NO races detected (${#tests_without_races_regular[@]}):${NC}"
        for test in "${tests_without_races_regular[@]}"; do
            echo -e "  ${YELLOW}✗${NC} $test"
        done
        echo ""
    fi
    
    # Tests with unknown status
    if [ ${#TESTS_UNKNOWN[@]} -gt 0 ]; then
        echo -e "${BLUE}? Tests with unknown race status (${#TESTS_UNKNOWN[@]}):${NC}"
        for test in "${TESTS_UNKNOWN[@]}"; do
            echo -e "  ${BLUE}?${NC} $test"
        done
        echo ""
    fi
    
    # Tests killed due to timeout
    if [ ${#TESTS_KILLED[@]} -gt 0 ]; then
        echo -e "${RED}⚠ Tests killed due to timeout (${#TESTS_KILLED[@]}):${NC}"
        for test in "${TESTS_KILLED[@]}"; do
            echo -e "  ${RED}⚠${NC} $test (timeout after 600 seconds)"
        done
        echo ""
    fi
    
    # False positive check tests (correct behavior = no race detected)
    if [ ${#FALSE_POSITIVE_TESTS_CORRECT[@]} -gt 0 ] || [ ${#FALSE_POSITIVE_TESTS_FAILED[@]} -gt 0 ]; then
        echo -e "${CYAN}━━━ False Positive Check Tests (should NOT detect races) ━━━${NC}"
        echo ""
        
        if [ ${#FALSE_POSITIVE_TESTS_CORRECT[@]} -gt 0 ]; then
            echo -e "${GREEN}✓ Correct - No false positives (${#FALSE_POSITIVE_TESTS_CORRECT[@]}):${NC}"
            for test in "${FALSE_POSITIVE_TESTS_CORRECT[@]}"; do
                echo -e "  ${GREEN}✓${NC} $test"
            done
            echo ""
        fi
        
        if [ ${#FALSE_POSITIVE_TESTS_FAILED[@]} -gt 0 ]; then
            echo -e "${RED}✗ FALSE POSITIVES - Races incorrectly detected (${#FALSE_POSITIVE_TESTS_FAILED[@]}):${NC}"
            for test in "${FALSE_POSITIVE_TESTS_FAILED[@]}"; do
                echo -e "  ${RED}✗${NC} $test"
            done
            echo ""
        fi
    fi
    
    # Summary
    echo -e "${CYAN}────────────────────────────────────────────────────────${NC}"
    echo -e "Total tests run: $total_tests"
    echo -e "False positive checks: ${CYAN}$((${#FALSE_POSITIVE_TESTS_CORRECT[@]} + ${#FALSE_POSITIVE_TESTS_FAILED[@]}))${NC}"
    echo ""
    echo -e "Races detected: ${GREEN}${#TESTS_WITH_RACES[@]}${NC}"
    echo -e "No races detected: ${YELLOW}${#tests_without_races_regular[@]}${NC}"
    if [ ${#TESTS_UNKNOWN[@]} -gt 0 ]; then
        echo -e "Unknown status: ${BLUE}${#TESTS_UNKNOWN[@]}${NC}"
    fi
    if [ ${#TESTS_KILLED[@]} -gt 0 ]; then
        echo -e "Killed (timeout): ${RED}${#TESTS_KILLED[@]}${NC}"
    fi
    
    # Detection rate (excluding false positive checks and killed tests from denominator)
    total_regular_tests=$((${#TESTS_WITH_RACES[@]} + ${#tests_without_races_regular[@]}))
    if [ $total_regular_tests -gt 0 ]; then
        detection_rate=$((${#TESTS_WITH_RACES[@]} * 100 / total_regular_tests))
        echo -e "\nDetection rate: ${detection_rate}% (${#TESTS_WITH_RACES[@]}/${total_regular_tests} tests)"
        echo -e "(Excluding false positive checks and killed tests from denominator)"
    fi
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
    echo ""
    fi
fi

echo -e "${GREEN}Done!${NC}"
