# Pagination

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Cursor pagination

All list endpoints use cursor-based pagination. Pass the limit query parameter to choose a page size; the default is 50 and the maximum is 200. Each response includes a next_cursor field. Send it back as the cursor parameter to fetch the following page.

## End of results

When there are no more results, next_cursor is null. Cursors are opaque strings and expire after 10 minutes, so do not store them for later use. Offset-based pagination is not supported.
