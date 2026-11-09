# Idempotent Requests

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Idempotency keys

To safely retry a POST request without creating duplicates, send an Idempotency-Key header containing a unique value such as a UUID. If the same key is sent again, the API returns the stored response from the first request instead of performing the action twice.

## Key retention

Idempotency keys are stored for 24 hours. After that, a repeated key is treated as a new request. Reusing a key with a different request body returns a 409 conflict error. GET and DELETE requests are already idempotent and ignore the header.
