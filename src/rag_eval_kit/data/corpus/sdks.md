# Official SDKs

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Languages

Larkspur maintains official client libraries for Python, Node.js, and Go. The Python SDK requires Python 3.9 or newer and is installed with pip install larkspur. The Node.js SDK supports Node 18 and newer. Community libraries exist for other languages but are not supported by Larkspur.

## Built-in behaviour

The official SDKs automatically retry 429 and 5xx responses up to 3 times with exponential backoff, attach an Idempotency-Key to POST requests, and follow next_cursor when you iterate over list results.
