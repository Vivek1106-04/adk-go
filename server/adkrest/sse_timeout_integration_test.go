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

package adkrest_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// TestRunSSEWithoutSSEWriteTimeout checks that /run_sse works on a server
// built without ServerConfig.SSEWriteTimeout. A zero timeout used to set the
// write deadline to the moment the request arrived, so every stream failed
// before its first byte.
func TestRunSSEWithoutSSEWriteTimeout(t *testing.T) {
	srv := httptest.NewServer(newCompactionServer(t, &echoModel{}, session.InMemoryService(), nil))
	defer srv.Close()
	sid := createCompactionSession(t, srv.URL)

	body, err := json.Marshal(map[string]any{
		"appName":    compactionApp,
		"userId":     compactionUser,
		"sessionId":  sid,
		"newMessage": genai.NewContentFromText("hi", genai.RoleUser),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(srv.URL+"/run_sse", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /run_sse error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading /run_sse body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /run_sse status = %d, want %d; body: %s", resp.StatusCode, http.StatusOK, got)
	}
	if !strings.Contains(string(got), "answer 1") {
		t.Errorf("POST /run_sse body does not contain the model's answer:\n%s", got)
	}
}
