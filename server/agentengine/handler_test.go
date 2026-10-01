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

package agentengine_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/server/agentengine"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestNewHandlerRejectsUnusableCompaction checks that Agent Engine refuses a
// compaction config it cannot serve, at construction.
//
// The config is validated inside runner.New, and this surface builds a runner
// per request, so without a check here the handler is created, the process
// reports healthy, and every request fails with the same error instead.
//
// This pins that the check runs, not how. NewHandler delegates to
// launcher.Config.Validate rather than reaching for the compaction field, so
// that a check added there later reaches this surface too, but a hand-rolled
// copy of today's check would satisfy this test just as well.
func TestNewHandlerRejectsUnusableCompaction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  *compaction.Config
		ok   bool
	}{
		{name: "nil compaction is fine", cfg: nil, ok: true},
		{name: "overlap with no interval", cfg: &compaction.Config{OverlapSize: 2}},
		{name: "no strategy at all", cfg: &compaction.Config{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg := &launcher.Config{
				SessionService: session.InMemoryService(),
				Compaction:     tc.cfg,
			}
			_, err := agentengine.NewHandler(cfg, time.Second, 1<<20, "engine")
			if tc.ok {
				if err != nil {
					t.Errorf("NewHandler() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("NewHandler() accepted a compaction config it cannot serve")
			}
			if !strings.Contains(err.Error(), "Compaction") {
				t.Errorf("error %q does not name the field an operator has to change", err)
			}
		})
	}
}

// TestQueryWithoutSSEWriteTimeout checks that a handler built with a zero
// sseWriteTimeout still answers. A zero timeout used to set the write deadline
// to the moment the request arrived, so every response was lost.
func TestQueryWithoutSSEWriteTimeout(t *testing.T) {
	h, err := agentengine.NewHandler(&launcher.Config{SessionService: session.InMemoryService()}, 0, 1<<20, "engine")
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	srv := httptest.NewServer(h)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/reasoning_engine", "application/json",
		strings.NewReader(`{"class_method":"async_create_session","input":{"user_id":"u"}}`))
	if err != nil {
		t.Fatalf("POST /reasoning_engine error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /reasoning_engine status = %d, want %d; body: %s", resp.StatusCode, http.StatusOK, got)
	}
	if !strings.Contains(string(got), `"user_id":"u"`) {
		t.Errorf("response does not contain the created session:\n%s", got)
	}
}
