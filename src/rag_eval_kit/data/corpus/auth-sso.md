# Single Sign-On (SAML)

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Availability

SAML single sign-on is available on the Business and Enterprise plans. It is not available on the Free or Team plans. Larkspur has tested setup guides for Okta, Microsoft Entra ID, and Google Workspace, and any SAML 2.0 identity provider should work.

## Enforcing SSO

Owners can enforce SSO so that members must sign in through the identity provider instead of using a password. Owners always keep a password fallback so they cannot be locked out by a misconfigured identity provider.

## Just-in-time provisioning

With just-in-time provisioning enabled, a Larkspur account is created automatically the first time someone from a verified domain signs in through SSO. New accounts created this way get the Member role.
