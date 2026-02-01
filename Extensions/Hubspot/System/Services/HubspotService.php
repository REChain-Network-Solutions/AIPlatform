<?php

namespace App\Extensions\Hubspot\System\Services;

use HubSpot\Client\Crm\Contacts\ApiException;
use HubSpot\Client\Crm\Contacts\Model\SimplePublicObjectInput;
use HubSpot\Discovery\Discovery;
use HubSpot\Factory;

class HubspotService
{
    private Discovery $hubspot;

    public function __construct()
    {
        $accessToken = setting('hubspot_access_token');
        $this->hubspot = Factory::createWithAccessToken($accessToken);
    }

    public function getCrmContacts(): array|string
    {
        try {
            return $this->hubspot->crm()->contacts()->basicApi()->getPage();
        } catch (ApiException $e) {
            return [
                'status'  => 'error',
                'code'    => $e->getCode(),
                'message' => $e->getMessage(),
                'details' => json_decode($e->getResponseBody(), true),
            ];
        }
    }

    public function createCrmContacts($email, $name, $surname): array|string
    {
        $contactInput = new SimplePublicObjectInput([
            'properties' => [
                'email'     => $email,
                'firstname' => $name,
                'lastname'  => $surname,
            ],
        ]);

        try {
            return $this->hubspot->crm()->contacts()->basicApi()->create($contactInput);
        } catch (ApiException $e) {
            return [
                'status'  => 'error',
                'code'    => $e->getCode(),
                'message' => $e->getMessage(),
                'details' => json_decode($e->getResponseBody(), true),
            ];
        }
    }
}
