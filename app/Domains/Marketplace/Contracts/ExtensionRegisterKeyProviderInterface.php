<?php

namespace App\Domains\Marketplace\Contracts;

interface ExtensionRegisterKeyProviderInterface
{
    public function registerKey(): string;
}
