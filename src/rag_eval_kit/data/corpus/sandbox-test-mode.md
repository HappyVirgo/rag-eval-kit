# Test Mode

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Sandbox data

Requests made with a test key operate on a separate sandbox copy of the workspace. Sandbox data never mixes with live data and is reset automatically every 30 days.

## Limits in test mode

Test mode has a fixed rate limit of 25 requests per minute regardless of plan. Webhooks fire in test mode too, and deliveries are marked with "livemode": false in the payload.

## Moving to production

To go live, replace the test key with a live key and register your production webhook endpoints separately; test-mode endpoints do not receive live events.
