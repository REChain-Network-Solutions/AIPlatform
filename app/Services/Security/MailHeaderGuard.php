<?php

declare(strict_types=1);

namespace App\Services\Security;

use RuntimeException;
use Symfony\Component\Mime\Address;
use Symfony\Component\Mime\Email;

class MailHeaderGuard
{
    public function assertSafe(Email $message): void
    {
        $this->assertText($message->getSubject() ?? '');

        foreach (array_merge(
            $message->getFrom(),
            $message->getTo(),
            $message->getCc(),
            $message->getBcc(),
            $message->getReplyTo(),
        ) as $address) {
            $this->assertAddress($address);
        }

        foreach ($message->getHeaders()->all() as $header) {
            $this->assertText($header->getName());
            if (method_exists($header, 'getBodyAsString')) {
                $this->assertText($header->getBodyAsString());
            }
        }
    }

    private function assertAddress(Address $address): void
    {
        $this->assertText($address->getAddress());
        $this->assertText($address->getName());
    }

    private function assertText(string $value): void
    {
        if (preg_match('/[\x00-\x1F\x7F]/u', $value)) {
            throw new RuntimeException('Mail header values may not contain control characters.');
        }
    }
}
