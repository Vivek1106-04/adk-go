// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agenttool_test

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type transferArgs struct {
	Amount int `json:"amount"`
}

func TestAgentTool_Run_PendingConfirmationReturnsError(t *testing.T) {
	ran := false
	transfer, err := functiontool.New(functiontool.Config{
		Name:                "transfer",
		Description:         "Transfers money.",
		RequireConfirmation: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		ran = true
		return map[string]any{"status": "done"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "call-1",
					Name: "transfer",
					Args: map[string]any{"amount": 100},
				}}},
			},
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "payments_agent",
		Description: "Moves money.",
		Model:       testLLM,
		Tools:       []tool.Tool{transfer},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "send 100"})

	if ran {
		t.Fatal("transfer ran without confirmation")
	}
	if err == nil {
		t.Fatalf("Run() = %v, nil; want an error for the unresolved confirmation", result)
	}
	if !strings.Contains(err.Error(), "unresolved long-running tool calls") {
		t.Errorf("Run() error = %q, want it to report the unresolved call", err)
	}
}

func TestAgentTool_Run_AnsweredLongRunningCallSucceeds(t *testing.T) {
	start, err := functiontool.New(functiontool.Config{
		Name:          "start_job",
		Description:   "Starts a job.",
		IsLongRunning: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		return map[string]any{"status": "started"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "call-1",
					Name: "start_job",
					Args: map[string]any{"amount": 1},
				}}},
			},
			genai.NewContentFromText("job started", genai.RoleModel),
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "jobs_agent",
		Description: "Starts jobs.",
		Model:       testLLM,
		Tools:       []tool.Tool{start},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "start"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	if got := result["result"]; got != "job started" {
		t.Errorf("Run() result = %v, want job started", result)
	}
}
