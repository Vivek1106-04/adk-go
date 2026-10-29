// Copyright 2025 Google LLC
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

package mcptoolset

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/semaphore"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/internal/version"
)

// MCPClient abstracts MCP session operations for easier connection management.
type MCPClient interface {
	CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error)
	ListTools(context.Context) ([]*mcp.Tool, error)
}

// connectionRefresher wraps an MCP client/transport and handles automatic reconnection.
// It implements MCPClient and transparently retries operations after reconnecting
// when the underlying session fails.
type connectionRefresher struct {
	client    *mcp.Client
	transport mcp.Transport

	// Session initialization and refresh perform network I/O. Waiters must be
	// able to leave when their own context ends without interrupting the owner.
	mu      *semaphore.Weighted
	session *mcp.ClientSession
	closed  bool

	// identityOnlyConnect connects sessions on identityOnlyContext(ctx)
	// rather than ctx.
	identityOnlyConnect bool

	closeOnce sync.Once
	closeErr  error
}

// refreshableErrors is a list of errors that should trigger a connection refresh.
var refreshableErrors = []error{
	mcp.ErrConnectionClosed,
	mcp.ErrSessionMissing,
	io.ErrClosedPipe,
	io.EOF,
}

// newConnectionRefresher creates a new connectionRefresher with the given client and transport.
// If client is nil, a default MCP client will be created.
func newConnectionRefresher(client *mcp.Client, transport mcp.Transport) *connectionRefresher {
	return &connectionRefresher{
		client:    orDefaultClient(client),
		transport: transport,
		mu:        semaphore.NewWeighted(1),
	}
}

// orDefaultClient returns client, or a default MCP client if it is nil.
func orDefaultClient(client *mcp.Client) *mcp.Client {
	if client == nil {
		return mcp.NewClient(&mcp.Implementation{Name: "adk-mcp-client", Version: version.Version}, nil)
	}
	return client
}

func (c *connectionRefresher) Close() error {
	c.closeOnce.Do(func() {
		// Close has no caller context, so it waits for any connection attempt
		// in progress. Acquire cannot fail with a context that is never done.
		_ = c.mu.Acquire(context.Background(), 1)
		// Closing is final so failed in-flight calls cannot reconnect.
		c.closed = true
		session := c.session
		c.session = nil
		c.mu.Release(1)

		if session != nil {
			c.closeErr = session.Close()
		}
	})
	return c.closeErr
}

// CallTool calls a tool on the MCP server, automatically reconnecting if needed.
func (c *connectionRefresher) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	result, _, err := withRetry(ctx, c, func(session *mcp.ClientSession) (*mcp.CallToolResult, error) {
		return session.CallTool(ctx, params)
	})
	return result, err
}

// ListTools lists all available tools from the MCP server, handling pagination
// and automatically reconnecting if needed. Per MCP spec, cursors do not persist
// across sessions, so pagination restarts from scratch after reconnection.
func (c *connectionRefresher) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	var tools []*mcp.Tool
	cursor := ""
	hasReconnected := false

	for {
		resp, reconnected, err := withRetry(ctx, c, func(session *mcp.ClientSession) (*mcp.ListToolsResult, error) {
			return session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list MCP tools: %w", err)
		}
		if reconnected {
			if hasReconnected {
				return nil, fmt.Errorf("failed to list MCP tools: connection lost again after reconnection")
			}
			// On reconnection, restart pagination from scratch per MCP spec.
			hasReconnected = true
			cursor = ""
			tools = nil
			continue
		}

		tools = append(tools, resp.Tools...)

		if resp.NextCursor == "" {
			break
		}
		cursor = resp.NextCursor
	}

	return tools, nil
}

// withRetry executes fn with the current session, and if it fails, attempts to refresh
// the connection and retry once. Returns the result, whether a reconnection occurred, and any error.
func withRetry[T any](ctx context.Context, c *connectionRefresher, fn func(*mcp.ClientSession) (T, error)) (T, bool, error) {
	var zero T

	session, err := c.getSession(ctx)
	if err != nil {
		return zero, false, err
	}

	result, err := fn(session)
	if err != nil {
		if !shouldRefreshConnection(err) {
			return zero, false, err
		}
		session, refreshErr := c.refreshConnection(ctx)
		if refreshErr != nil {
			return zero, false, fmt.Errorf("%w (reconnection also failed: %v)", err, refreshErr)
		}
		result, err = fn(session)
		return result, true, err
	}
	return result, false, err
}

// shouldRefreshConnection returns true if the error indicates we should
// attempt to refresh the MCP connection.
func shouldRefreshConnection(err error) bool {
	for _, target := range refreshableErrors {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
}

func (c *connectionRefresher) getSession(ctx context.Context) (*mcp.ClientSession, error) {
	if err := c.mu.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer c.mu.Release(1)

	if c.closed {
		return nil, mcp.ErrConnectionClosed
	}
	if c.session != nil {
		return c.session, nil
	}

	session, err := c.connect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to init MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

func (c *connectionRefresher) refreshConnection(ctx context.Context) (*mcp.ClientSession, error) {
	if err := c.mu.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer c.mu.Release(1)

	if c.closed {
		return nil, mcp.ErrConnectionClosed
	}

	// Ping to verify the connection is actually dead before reconnecting.
	// This handles the case where another goroutine already reconnected.
	if c.session != nil {
		if err := c.session.Ping(ctx, &mcp.PingParams{}); err == nil {
			return c.session, nil
		}
		if err := c.session.Close(); err != nil {
			log.Printf("failed to close MCP session: %v", err) //nolint:forbidigo // pre-slog call site
		}
		c.session = nil
	}

	session, err := c.connect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to refresh MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

// connect opens a new MCP session. The caller holds c.mu.
func (c *connectionRefresher) connect(ctx context.Context) (*mcp.ClientSession, error) {
	if !c.identityOnlyConnect {
		return c.client.Connect(ctx, c.transport, nil)
	}
	connCtx, stop := identityOnlyContext(ctx)
	defer stop()
	return c.client.Connect(connCtx, c.transport, nil)
}

// disconnected reports whether c has no session and no connection attempt
// in progress.
func (c *connectionRefresher) disconnected() bool {
	if !c.mu.TryAcquire(1) {
		return false
	}
	defer c.mu.Release(1)
	return c.session == nil
}

var _ MCPClient = (*connectionRefresher)(nil)

// Bounds on the sessions a credentialSessions keeps. Variables so tests can
// shrink them.
var (
	// sessionIdleTimeout is how long a session may go unused before it is
	// closed.
	sessionIdleTimeout = 10 * time.Minute
	// maxSessions caps the sessions kept open. Sessions in use are never
	// evicted, so the cap is exceeded only while more than that many are busy.
	maxSessions = 64
)

// credentialSessions gives each distinct credential an MCP session of its own.
//
// It is used when Config.Auth is set. An MCP server that binds a session to
// the user who created it (as the MCP spec recommends, and as go-sdk servers do
// whenever their token verifier reports a UserID) rejects requests made on that
// session with another user's credential, and the rejection also closes the
// session for its owner. So a provider that mints a credential per user needs a
// session per user, while a user-independent one such as [auth.StaticToken]
// serves every caller on one session. Keying sessions on the credential the
// provider resolves covers both, as adk-python does with its header-based
// session key.
//
// A session is closed once it has been idle for sessionIdleTimeout, or when
// the pool is full and it is the least recently used idle one.
type credentialSessions struct {
	client    *mcp.Client
	transport mcp.Transport
	provider  auth.CredentialProvider

	mu       sync.Mutex
	closed   bool
	sessions map[string]*pooledSession
	// evicting tracks evicted sessions that are still closing, so Close can
	// wait for them.
	evicting sync.WaitGroup

	closeOnce sync.Once
	closeErr  error
}

// pooledSession is one credential's connection and its use. inUse and lastUsed
// are guarded by credentialSessions.mu.
type pooledSession struct {
	key      string
	conn     *connectionRefresher
	inUse    int
	lastUsed time.Time
}

func newCredentialSessions(client *mcp.Client, transport mcp.Transport, provider auth.CredentialProvider) *credentialSessions {
	return &credentialSessions{
		client:    orDefaultClient(client),
		transport: transport,
		provider:  provider,
		sessions:  make(map[string]*pooledSession),
	}
}

// sessionKey returns a digest of the headers the provider's credential sets
// for the caller in ctx.
func (p *credentialSessions) sessionKey(ctx context.Context) (string, error) {
	cred, err := p.provider.Credential(ctx)
	if err != nil {
		return "", fmt.Errorf("auth: resolve credential: %w", err)
	}
	if cred == nil {
		return "", fmt.Errorf("auth: provider returned nil credential")
	}
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		return "", fmt.Errorf("auth: apply credential: %w", err)
	}
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	slices.Sort(names)
	// Length-prefix every field so that no two header sets encode alike.
	var b []byte
	for _, name := range names {
		b = fmt.Appendf(b, "%d:%s%d:", len(name), name, len(h[name]))
		for _, v := range h[name] {
			b = fmt.Appendf(b, "%d:%s", len(v), v)
		}
	}
	sum := sha256.Sum256(b)
	return string(sum[:]), nil
}

// acquire returns the session for the credential of the caller in ctx,
// creating it on first use, and marks it in use until release. The session
// itself connects lazily.
func (p *credentialSessions) acquire(ctx context.Context) (*pooledSession, error) {
	if p.isClosed() {
		return nil, mcp.ErrConnectionClosed
	}
	key, err := p.sessionKey(ctx)
	if err != nil {
		return nil, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, mcp.ErrConnectionClosed
	}
	now := time.Now()
	s, ok := p.sessions[key]
	p.evictLocked(now, !ok)
	if !ok {
		conn := newConnectionRefresher(p.client, p.transport)
		conn.identityOnlyConnect = true
		s = &pooledSession{key: key, conn: conn}
		p.sessions[key] = s
	}
	s.inUse++
	s.lastUsed = now
	return s, nil
}

func (p *credentialSessions) isClosed() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closed
}

// release ends a use of s that finished with err. A session that never
// connected, or lost its connection and could not reconnect, is dropped so the
// pool does not keep entries for credentials the server refuses.
func (p *credentialSessions) release(s *pooledSession, err error) {
	drop := err != nil && s.conn.disconnected()

	p.mu.Lock()
	s.inUse--
	s.lastUsed = time.Now()
	if drop && s.inUse == 0 && p.sessions[s.key] == s {
		p.evictSessionLocked(s)
	}
	p.mu.Unlock()
}

// evictLocked closes idle sessions unused for sessionIdleTimeout and, when
// makeRoom is set and the pool is full, the least recently used idle session.
func (p *credentialSessions) evictLocked(now time.Time, makeRoom bool) {
	var lru *pooledSession
	for _, s := range p.sessions {
		if s.inUse > 0 {
			continue
		}
		if now.Sub(s.lastUsed) >= sessionIdleTimeout {
			p.evictSessionLocked(s)
			continue
		}
		if lru == nil || s.lastUsed.Before(lru.lastUsed) {
			lru = s
		}
	}
	if makeRoom && len(p.sessions) >= maxSessions && lru != nil {
		p.evictSessionLocked(lru)
	}
}

// evictSessionLocked removes s from the pool and closes it in the background,
// since closing sends a request to the server.
func (p *credentialSessions) evictSessionLocked(s *pooledSession) {
	delete(p.sessions, s.key)
	p.evicting.Go(func() {
		// No caller is left to report a failure to, and the session is gone
		// from the pool either way.
		_ = s.conn.Close()
	})
}

// CallTool calls a tool on the session for the caller's credential.
func (p *credentialSessions) CallTool(ctx context.Context, params *mcp.CallToolParams) (result *mcp.CallToolResult, err error) {
	s, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.release(s, err) }()
	return s.conn.CallTool(ctx, params)
}

// ListTools lists tools on the session for the caller's credential.
func (p *credentialSessions) ListTools(ctx context.Context) (tools []*mcp.Tool, err error) {
	s, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.release(s, err) }()
	return s.conn.ListTools(ctx)
}

// Close closes every session, waiting for connection setup and MCP operations
// in progress, as [connectionRefresher.Close] does. It is safe to call
// concurrently or more than once, and every call returns the first one's
// result. Later operations fail with [mcp.ErrConnectionClosed].
func (p *credentialSessions) Close() error {
	p.closeOnce.Do(func() {
		p.mu.Lock()
		p.closed = true
		sessions := p.sessions
		p.sessions = nil
		p.mu.Unlock()

		var errs []error
		for _, s := range sessions {
			if err := s.conn.Close(); err != nil {
				errs = append(errs, err)
			}
		}
		p.evicting.Wait()
		p.closeErr = errors.Join(errs...)
	})
	return p.closeErr
}

var _ MCPClient = (*credentialSessions)(nil)

// identityOnlyContext returns a context that carries the ADK identity of ctx
// and no other value, and is canceled when ctx is until stop is called.
//
// go-sdk keeps the context a streamable session is connected on, minus its
// cancellation, for the life of the session, so that the standalone stream and
// the closing request carry the same values. Connecting on an invocation's
// context would keep that whole invocation and its session in memory until the
// MCP session closes. The identity is all the credential provider needs.
func identityOnlyContext(ctx context.Context) (connCtx context.Context, stop func()) {
	base := context.Background()
	if id, ok := agent.IdentityFromContext(ctx); ok {
		base = context.WithValue(base, adkcontext.IdentityKey, id)
	}
	connCtx, cancel := context.WithCancelCause(base)
	stopCancel := context.AfterFunc(ctx, func() { cancel(context.Cause(ctx)) })
	return connCtx, func() {
		stopCancel()
		cancel(nil)
	}
}
