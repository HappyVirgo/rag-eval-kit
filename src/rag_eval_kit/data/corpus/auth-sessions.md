# Sessions

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Session lifetime

Web sessions expire after 14 days of inactivity. Activity on any device refreshes only that device's session. Mobile app sessions last 90 days.

## Concurrent sessions

A single user can have at most 10 concurrent web sessions. When an eleventh session is created, the oldest session is signed out automatically.

## Revoking sessions

Users can sign out of all other devices from Profile > Security. Admins can revoke every session for a member from the member's profile page, which is useful when a laptop is lost.
