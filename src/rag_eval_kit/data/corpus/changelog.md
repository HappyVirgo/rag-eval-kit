# Changelog

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## 2025-03-01 API version

Raised the maximum page size for list endpoints from 100 to 200. Removed the offset query parameter from all list endpoints. Added the task.completed webhook event. Errors now include a request_id field.

## 2024-09-15

Webhook deliveries are now retried for longer than before: the previous schedule of 5 attempts over 1 hour was replaced with a longer exponential schedule. Added WebAuthn security keys as a second factor.

## 2024-04-02

Introduced the Business plan between Team and Enterprise. Before this change, SAML single sign-on was an Enterprise-only feature. The legacy Starter plan stopped accepting new customers.
