# Error Handling

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Error format

Errors are returned as JSON objects with three fields: code, message, and request_id. The request_id is unique per request; include it when you contact support so engineers can find the request in the logs.

## Common error codes

A 400 response with code invalid_request means a parameter is missing or malformed. A 401 response with code unauthenticated means the API key or token is missing or invalid. A 403 response with code forbidden means the key is valid but lacks permission. A 404 response with code not_found means the resource does not exist or is not visible to the caller. A 409 response with code conflict usually means an idempotency key was reused with a different request body.

## Server errors

A 500 or 503 response means something went wrong on the Larkspur side. These requests are safe to retry with exponential backoff.
