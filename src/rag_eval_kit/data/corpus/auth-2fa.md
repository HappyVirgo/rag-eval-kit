# Two-Factor Authentication

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Methods

Larkspur supports two-factor authentication with authenticator apps that generate time-based one-time passwords (TOTP) and with WebAuthn security keys such as hardware keys and passkeys. SMS codes are not offered.

## Recovery codes

When you turn on 2FA you receive 10 single-use recovery codes. Each code can be used once if you lose access to your authenticator. You can generate a fresh set at any time, which invalidates the old codes.

## Requiring 2FA

Admins on any paid plan can require two-factor authentication for every member of the workspace. Members who have not set it up are asked to enroll the next time they sign in.
