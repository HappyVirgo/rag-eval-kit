# Audit Logs

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## What is logged

Audit logs record security-relevant events such as sign-ins, role changes, API key creation, SSO configuration changes, and exports. Each entry contains the actor, the action, the IP address, and a timestamp.

## Availability and retention

Audit logs are available on the Business and Enterprise plans. Business workspaces keep audit logs for 1 year. Enterprise workspaces keep them for 7 years.

## Audit log API

Admins can read audit events with GET /v1/audit-events, which supports filtering by actor and date range. The endpoint uses the same cursor pagination as other list endpoints.
