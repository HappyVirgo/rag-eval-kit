# Migrating to API Version 2025-03-01

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Pagination changes

Version 2025-03-01 removes offset pagination. Replace offset and page parameters with the cursor parameter and read next_cursor from each response until it is null.

## Renamed fields

The task field due is renamed to due_at, and the field owner is renamed to assignee_id. The old names are still returned by earlier versions.

## Testing the upgrade

Send the Larkspur-Version: 2025-03-01 header from a test key first and run your integration tests against the sandbox before switching your live keys.
