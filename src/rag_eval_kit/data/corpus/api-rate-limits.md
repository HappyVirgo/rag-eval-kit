# API Rate Limits

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Limits by plan

Rate limits are applied per API key and measured in requests per minute. The Free plan allows 60 requests per minute, the Team plan allows 300 requests per minute, and the Business plan allows 1,000 requests per minute. Enterprise limits are set in the contract and start at 3,000 requests per minute.

## Rate limit headers

Every response includes the headers X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset. The reset header is a Unix timestamp in seconds that marks when the current window ends.

## Exceeding the limit

When you exceed the limit the API returns HTTP 429 Too Many Requests with a Retry-After header that says how many seconds to wait. Clients should back off exponentially instead of retrying immediately. Bulk endpoints count each item in the batch as one request.
