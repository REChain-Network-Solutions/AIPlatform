<?php

declare(strict_types=1);

namespace App\Events;

use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

/**
 * Fired by any extension when a contact is captured (created or enriched with
 * an email/phone). The CRM extension listens for this to mirror the contact
 * into crm_contacts. Inert when CRM is not installed (no listener registered).
 */
class ContactCapturedEvent
{
    use Dispatchable;
    use SerializesModels;

    public function __construct(
        public int $userId,
        public string $name,
        public ?string $email = null,
        public ?string $phone = null,
        public ?int $countryCode = null,
        public string $source = 'unknown',
    ) {}
}
