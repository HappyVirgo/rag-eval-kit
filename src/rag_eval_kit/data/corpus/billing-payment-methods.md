# Payment Methods and Failed Payments

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Supported payment methods

Larkspur accepts Visa, Mastercard, and American Express cards. ACH bank debit is available for workspaces on the Business and Enterprise plans. PayPal and cryptocurrency are not accepted.

## Failed payments

If a card payment fails, Larkspur retries the charge three times over seven days. Workspace owners receive an email after each failed attempt. If all retries fail, the workspace becomes read-only: members can view and export data but cannot create or edit tasks until a valid payment method is added.
