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

# cuHadron Makefile

# CUDA toolkit root; default /usr/local/cuda. Override if your install is elsewhere (e.g. export CUDA_HOME=/opt/cuda).
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

# Race injection control: FIXED=0 (default) = race present, FIXED=1 = synchronized (no race)
FIXED ?= 0

# Target architecture. Binary suffix has no underscore (sm90, sm100). No global minimum; see test_requirements.txt per category.
ARCH ?= sm_90
ARCH_SUFFIX = $(subst _,,$(ARCH))
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))

# Per-category requirements (min_arch, min_gpus). Only build categories that support current ARCH.
REQUIREMENTS_FILE = test_requirements.txt
ALLOWED_CATEGORIES = $(shell awk -v arch='$(ARCH_NUM)' 'NF>=2 && $$1 !~ /^#/ && ($$2+0 == 0 || $$2+0 <= arch+0) { print $$1 }' $(REQUIREMENTS_FILE) 2>/dev/null)
# Sources: only .cu files in categories that are allowed for this ARCH
SRCS = $(foreach cat,$(ALLOWED_CATEGORIES),$(wildcard $(cat)/*.cu))

# Generate binary names
BINS = $(SRCS:.cu=.$(ARCH_SUFFIX).out)

.PHONY: all clean help

all: $(BINS)

# Race control flags
ifeq ($(FIXED),1)
    RACE_FLAGS = -DFIXED
else
    RACE_FLAGS =
endif

%.$(ARCH_SUFFIX).out: %.cu
	@category=$(*D); \
	req=$$(grep "^$$category " $(REQUIREMENTS_FILE) 2>/dev/null | head -1); \
	min_arch=$$(echo $$req | awk '{print $$2}'); \
	min_gpus=$$(echo $$req | awk '{print $$3}'); \
	extra=""; \
	[ -n "$$min_arch" ] && [ "$$min_arch" -gt 0 ] 2>/dev/null && extra="$$extra -DREQUIRE_ARCH=$$min_arch"; \
	[ -n "$$min_gpus" ] && [ "$$min_gpus" -gt 1 ] 2>/dev/null && extra="$$extra -DREQUIRE_GPUS=$$min_gpus"; \
	$(NVCC) -arch=$(ARCH) -lineinfo $(RACE_FLAGS) $$extra $< -o $@

# Remove all built binaries (all architectures)
clean:
	rm -f */*.out

help:
	@echo "cuHadron"
	@echo ""
	@echo "  make              - Build all tests for current ARCH (default ARCH=sm_90)"
	@echo "  make ARCH=sm_80   - Build tests that support sm_80 (bulkcpy, dsmem excluded)"
	@echo "  make ARCH=sm_100  - Build for e.g. Blackwell"
	@echo "  make FIXED=1      - Build race-free versions"
	@echo "  make clean        - Remove all binaries"
	@echo ""
	@echo "  Per-category requirements (min_arch, min_gpus) are in test_requirements.txt."
	@echo "  Only tests whose category supports the chosen ARCH are built."
	@echo ""
	@echo "Race injection control:"
	@echo "  make FIXED=0      - Build with race (default)"
	@echo "  make FIXED=1      - Build with synchronization (no race)"
	@echo ""
	@echo "Binaries: <test>.$(ARCH_SUFFIX).out"
