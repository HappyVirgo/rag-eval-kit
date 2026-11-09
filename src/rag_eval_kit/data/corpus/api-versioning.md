# API Versioning

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Date-based versions

The Larkspur API uses date-based versions such as 2025-03-01. Pin a version by sending the Larkspur-Version header with each request. Requests without the header use the version that was current when the API key was created.

## Deprecation policy

A version is supported for at least 12 months after it is deprecated. Deprecation notices are announced in the changelog and emailed to workspace admins. Responses from a deprecated version include a Sunset header with the date the version will stop working.

## Breaking changes

Removing a field, renaming a field, or changing a field type is a breaking change and only happens in a new version. Adding new optional fields or new event types is not considered breaking.
