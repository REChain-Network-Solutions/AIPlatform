<?php

namespace App\Domains\Marketplace\Exceptions;

use Exception;
use Throwable;

class InvalidLicenseException extends Exception
{
    public function __construct(string $message = 'Invalid license key provided.', int $code = 403, ?Throwable $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }

    public function render(): string
    {
        return $this->getMessage();
    }
}
