# Webhook Delivery and Retries

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Successful delivery

Your endpoint must respond with a 2xx status code within 10 seconds. Any other status code, or a timeout, counts as a failed delivery.

## Retry schedule

Failed deliveries are retried with exponential backoff for up to 8 attempts spread over roughly 24 hours. Events may arrive out of order during retries, so use the created_at timestamp to order them.

## Disabled endpoints

If every delivery to an endpoint fails for 3 consecutive days, the endpoint is disabled automatically and workspace admins are notified by email. You can re-enable it from the dashboard and replay missed events from the last 7 days.
