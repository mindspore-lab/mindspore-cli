# Provider Import Duplicate Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show both the env-backed import option and the original provider option while preserving their different API key flows.

**Architecture:** Keep the provider catalog unchanged. Encode import selections with an internal option ID, decode that ID in the connect flow, and only allow detected env API key fallback for decoded import selections.

**Tech Stack:** Go, Bubble Tea UI tests, application workflow tests

---

### Task 1: Lock expected behavior with tests

**Files:**
- Modify: `internal/app/commands_model_test.go`
- Modify: `ui/app_model_browser_test.go`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run narrow Go tests to verify the failures are for the intended behavior**
- [ ] **Step 3: Keep test coverage focused on duplicate entries and import-vs-normal connect behavior**

### Task 2: Implement minimal option/source separation

**Files:**
- Modify: `internal/app/provider_workflow.go`

- [ ] **Step 1: Stop filtering catalog providers when import suggestions exist**
- [ ] **Step 2: Add internal import option ID encoding/decoding helpers**
- [ ] **Step 3: Route connect behavior through decoded import metadata**
- [ ] **Step 4: Only allow env API key fallback for import-origin selections**

### Task 3: Verify and clean up

**Files:**
- Modify: `internal/app/commands_model_test.go`
- Modify: `ui/app_model_browser_test.go`
- Modify: `internal/app/provider_workflow.go`

- [ ] **Step 1: Run the narrowest relevant test commands**
- [ ] **Step 2: Widen to related package tests if the narrow checks pass**
- [ ] **Step 3: Confirm no unintended behavior changes are required**
