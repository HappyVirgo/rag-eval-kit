# Verifying Webhook Signatures

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Signature header

Every webhook delivery includes a Larkspur-Signature header in the form t=timestamp,v1=signature. The signature is an HMAC-SHA256 of the timestamp, a period, and the raw request body, computed with the endpoint's signing secret.

## Verifying

Compute the expected HMAC on your server and compare it with the v1 value using a constant-time comparison. Reject deliveries whose timestamp is more than 5 minutes old to protect against replay attacks. Always verify against the raw body bytes, not a re-serialized JSON object.

## Signing secrets

Each endpoint has its own signing secret, which starts with whsec_. You can roll the secret from the endpoint settings page; the old secret stays valid for 24 hours.
