

## Auto Convert on Accept
To automatically convert accepted quotes into invoices (in the EInvoice module), set:
```
QUOTES_AUTO_CONVERT_ON_ACCEPT=true
```
If EInvoice isn't installed, acceptance still records status but no invoice is created.


## Staff Notifications + Webhook

Add to `.env`:

```
# Comma-separated staff emails to notify on acceptance
QUOTES_NOTIFY_EMAILS="ops@example.com,sales@example.com"

# Webhook to call on acceptance (optional)
QUOTES_WEBHOOK_ACCEPT_URL="https://example.com/webhooks/quotes"
QUOTES_WEBHOOK_SECRET="changeme"
```
The webhook receives JSON body like:
```json
{
  "event": "quote.accepted",
  "quote_id": 123,
  "quote_number": "Q-ABC123",
  "client_id": 45,
  "currency": "USD",
  "subtotal": 100.0,
  "tax_total": 10.0,
  "grand_total": 110.0,
  "accepted_at": "2025-09-29T11:22:33+00:00",
  "invoice_id": 789
}
```
Header: `X-Quotes-Signature: HMAC_SHA256(body, QUOTES_WEBHOOK_SECRET)`
