# Webhooks Overview

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## What webhooks do

Webhooks send an HTTP POST request to your server when something happens in a workspace. Supported events include task.created, task.updated, task.completed, project.archived, and member.invited.

## Endpoints

Each workspace can register up to 20 webhook endpoints. Endpoint URLs must use HTTPS; plain HTTP URLs are rejected. Each endpoint can subscribe to a subset of event types. Webhooks are available on every plan, including Free.

## Payloads

Each delivery contains an event id, the event type, a created_at timestamp, and a data object describing the resource. Event ids are unique, so use them to discard duplicate deliveries.
