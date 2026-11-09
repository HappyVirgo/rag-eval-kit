# API Keys

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Creating keys

Only workspace owners and admins can create API keys, from Settings > Developers > API keys. Live keys start with the prefix lsk_live_ and test keys start with lsk_test_. A key is displayed only once at creation time, so store it in a secrets manager immediately.

## Limits and rotation

Each workspace can have at most 25 active API keys. Keys do not expire automatically, but we recommend rotating them every 90 days. When you roll a key, the old key keeps working for 24 hours so you can deploy the new one without downtime.

## Using a key

Send the key in the Authorization header as a bearer token: Authorization: Bearer lsk_live_... Requests without a valid key receive a 401 response.
