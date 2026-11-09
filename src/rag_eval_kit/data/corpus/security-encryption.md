# Encryption and Compliance

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Encryption

All customer data is encrypted at rest with AES-256. Data in transit is protected with TLS 1.2 or newer; older protocol versions are rejected. Encryption keys are rotated every year.

## Customer-managed keys

Enterprise workspaces can bring their own encryption key stored in a supported cloud key management service. If the customer revokes the key, the workspace becomes unavailable until access is restored.

## Compliance reports

Larkspur publishes a SOC 2 Type II report every year. Customers on the Business and Enterprise plans can request the report under NDA from the trust center. This is a fictional statement about a fictional company.
