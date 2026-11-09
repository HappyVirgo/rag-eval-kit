# OAuth 2.0 Apps

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Authorization flow

Third-party apps that act on behalf of a user must use the OAuth 2.0 authorization code flow. PKCE is required for all apps, including server-side apps. The implicit flow is not supported.

## Token lifetimes

Access tokens expire after 1 hour. Refresh tokens are valid for 60 days and are rotated every time they are used: each refresh returns a new refresh token and invalidates the previous one.

## Scopes

Apps request scopes such as tasks:read, tasks:write, projects:read, and webhooks:manage. Users see the requested scopes on the consent screen and can revoke an app at any time from Settings > Connected apps.
